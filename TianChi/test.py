import pandas as pd
import os
import gc
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

#--------------------------数据读取--------------------------
train = pd.read_csv('train.csv')
testA = pd.read_csv('testA.csv')

# 调用read_csv()读取文件时时会自动识别表头（默认读取第一行，即header=0)，数据有表头时不能设置header为空；
train.head()

# --------------------------数据预处理--------------------------
# 将testA拼接到train的末端
data = pd.concat([train, testA], axis=0, ignore_index=True) 

# （可删除）统计grade及subGrade的数据
print(sorted(data['grade'].unique()))
print(sorted(data['subGrade'].unique()))

# （可删除）统计employmentLength共有多少个值，并统计重复次数，其中“dropna = False”表示保留NAN
data['employmentLength'].value_counts(dropna=False).sort_index()

# 根据统计结果将'10+ years'转换成'10 years'，将‘<1 year’转换成'0 years'
data['employmentLength'].replace(to_replace='10+ years', value='10 years', inplace=True)
data['employmentLength'].replace('< 1 year', '0 years', inplace=True)

# 用于提取employmentLength中的数字
def employmentLength_to_int(s):
    if pd.isnull(s):
        return s
    else:
        return np.int8(s.split()[0])

# DataFrame.apply() 函数则会遍历行或列的每一个元素，对元素运行指定的 function。
data['employmentLength'] = data['employmentLength'].apply(employmentLength_to_int)

# （可删除）查看处理后的'employmentLength'是否符合预期
data['employmentLength'].value_counts(dropna=False).sort_index()

# （可删除）对earliesCreditLine的数据进行随机抽样
data['earliesCreditLine'].sample(5)

# 将'earliesCreditLine'中的year转化成int型病代替原来的'month-year'
data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda s: int(s[-4:]))

# （可删除）describe()：返回 Series（一维）和DataFrame（二维）这两个pandas核心数据结构的统计变量（数字：count、mean、std、max、min、dtype...；字母：count、unique、top、freq、dtype...）
data['earliesCreditLine'].describe()

# 用于打印表头以及前五行数据来观察预处理后的数据是否准确（默认n=5）
print(data.head())

# 部分类别特征
cate_features = ['grade', 'subGrade', 'employmentTitle', 'homeOwnership', 'verificationStatus', 'purpose', 'postCode', 'regionCode', \
                 'applicationType', 'initialListStatus', 'title', 'policyCode']

# （可删除）nunique():用于获取唯一值的统计次数。
for f in cate_features:
    print(f, '类型数：', data[f].nunique())

# 类型数在2之上，又不是高维稀疏的，需要使用get_dummies实现one hot encode
# 也即：就是添加原来数据中没有的变量，但是这并不是意味着可以随意添加，应该是根据原来的数据进行转换。
# 例如：将一个变量Embarked，根据它的值（C、Q、S）转换为Embarked_C、Embarked_Q、Embarked_S三个变量
# (转化后有默认名，也可以利用参数prefix来自己修改）
data = pd.get_dummies(data, columns=['grade', 'subGrade', 'homeOwnership', 'verificationStatus', 'purpose', 'regionCode'], drop_first=True)

# 高维类别特征需要进行转换
for f in ['employmentTitle', 'postCode', 'title']:
    # 这个操作的目的是将数据按照${f}进行分组（因为将train表和testA表合并了）
    # 然后统计每组的个数（count），并且新增一列'f_cnts'的属性，其值为相应的count
    # tips：此操作并未将该列属性增加到csv表格中，如果需要则应使用set_index()
    data[f+'_cnts'] = data.groupby([f])['id'].transform('count')

    # 增加一组属性：对属性${f}进行降序（ascending = False）排，
    # 并且使用astype()将其数值强制转化成int
    data[f+'_rank'] = data.groupby([f])['id'].rank(ascending=False).astype(int)
    
    # 删除${f}这列属性
    del data[f]

# 将除了id、issueDate、isDefault外的所有属性作为Features进行训练
features = [f for f in data.columns if f not in ['id','issueDate','isDefault']]

# 训练组为属性isDefault有明确值的数据，测试组则与之相反
train = data[data.isDefault.notnull()].reset_index(drop=True)
test = data[data.isDefault.isnull()].reset_index(drop=True)

x_train = train[features]
x_test = test[features]
y_train = train['isDefault']


# --------------------------建立训练器-------------------------
# 构建一个函数能够直接调用「三种树模型」
def cv_model(clf, train_x, train_y, test_x, clf_name):
    folds = 5   
    seed = 2020 
    # KFold函数的作用：对数据进行交叉验证。
    # 原理是将数据分为"n_splits"组，每组数据都要作为一次验证集进行一次验证，而其余的"n_splits"组数据作为训练集。
    # 则一共要循环、验证"n_splits"次，得到"n_splits"个模型，将这些模型得到的误差计算均值，得到交叉验证误差。
    # n_splits： 将训练集分为n份数据
    # shuffle： 是否打乱数据的顺序，只有当其为true时random_state才能发挥其作用。
    # random_state:在需要设置随机数种子的地方都设置好，从而保证程序每次运行都分割一样的训练集和测试集。
    # 进而每次运行代码时都能得到同样的结果，否则，同样的算法模型在不同的训练集和测试集上的效果不一样。
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

    # shape[0]返回的是相应的行数，shape[1]则是返回列数
    # zeros函数的作用：返回一个给定维数的向量
    # 输入一个数字n，则返回1*n的向量；若输入一个元组(n,m)则返回n*m的向量
    train = np.zeros(train_x.shape[0])
    test = np.zeros(test_x.shape[0])

    cv_scores = []

    # split函数的作用：返回样本切分之后数据集的indices（索引）
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i+1)))
        # iloc函数的作用：通过行号（也即前面的索引）来取行数据
        # trn_x是交叉验证中的测试集的[feature]，trn_y是交叉验证中测试集的['isDefault']；val_则是测试集
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]

        # LightGBM算法：使用GOSS算法和EFB算法的梯度提升树（GBDT）称之为LightGBM
        if clf_name == "lgb":

            # 保存 Dataset 到 LightGBM 二进制文件将会使得加载更快速:
            train_matrix = clf.Dataset(trn_x, label=trn_y)
            valid_matrix = clf.Dataset(val_x, label=val_y)

            params = {
                'boosting_type': 'gbdt', # 设置提升类型
                'objective': 'binary',  # 逻辑回归二分类，输出概率
                'metric': 'auc',    # 度量指标AUC：Area Under the Curve 
                'min_child_weight': 5,  #使一个结点分裂的最小海森值之和，较高的值可能会减少过度拟合
                'num_leaves': 2 ** 5,   # 树的最大叶子节点数
                'lambda_l2': 10,    # 越小l2正则程度越高 
                'feature_fraction': 0.8, # 建树时随机抽取特征的比例，默认为1
                'bagging_fraction': 0.8, # 建树时样本采样比例，默认为1
                'bagging_freq': 4,  #bagging的频率
                'learning_rate': 0.1, #学习率
                'seed': 2020,   #随机数种子
                'nthread': 28,
                'n_jobs':24,
                'silent': True,
                'verbose': -1,
            }

            model = clf.train(params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], 
            verbose_eval=200,early_stopping_rounds=200)
            val_pred = model.predict(val_x, num_iteration=model.best_iteration)
            test_pred = model.predict(test_x, num_iteration=model.best_iteration)
            
            # print(list(sorted(zip(features, model.feature_importance("gain")), key=lambda x: x[1], reverse=True))[:20])
                
        if clf_name == "xgb":
            train_matrix = clf.DMatrix(trn_x , label=trn_y)
            valid_matrix = clf.DMatrix(val_x , label=val_y)
            test_matrix = clf.DMatrix(test_x)
            
            params = {'booster': 'gbtree',
                      'objective': 'binary:logistic',
                      'eval_metric': 'auc',
                      'gamma': 1,
                      'min_child_weight': 1.5,
                      'max_depth': 5,
                      'lambda': 10,
                      'subsample': 0.7,
                      'colsample_bytree': 0.7,
                      'colsample_bylevel': 0.7,
                      'eta': 0.04,
                      'tree_method': 'exact',
                      'seed': 2020,
                      'nthread': 36,
                      "silent": True,
                      }
            
            watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]
            
            model = clf.train(params, train_matrix, num_boost_round=50000, evals=watchlist, verbose_eval=200, early_stopping_rounds=200)
            val_pred  = model.predict(valid_matrix, ntree_limit=model.best_ntree_limit)
            test_pred = model.predict(test_matrix , ntree_limit=model.best_ntree_limit)
                 
        if clf_name == "cat":
            params = {
                    'learning_rate': 0.05, 
                    'depth': 5, 
                    'l2_leaf_reg': 10, 
                    'bootstrap_type': 'Bernoulli',
                    'od_type': 'Iter', 
                    'od_wait': 50, 
                    'random_seed': 11, 
                    'allow_writing_files': False
                    }
            
            model = clf(iterations=20000, **params)
            model.fit(trn_x, trn_y, eval_set=(val_x, val_y),
                      cat_features=[], use_best_model=True, verbose=500)
            
            val_pred  = model.predict(val_x)
            test_pred = model.predict(test_x)
            
        train[valid_index] = val_pred
        test = test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))
        
        print(cv_scores)
       
    print("%s_scotrainre_list:" % clf_name, cv_scores)
    print("%s_score_mean:" % clf_name, np.mean(cv_scores))
    print("%s_score_std:" % clf_name, np.std(cv_scores))
    return train, test

def lgb_model(x_train, y_train, x_test):
    lgb_train, lgb_test = cv_model(lgb, x_train, y_train, x_test, "lgb")
    return lgb_train, lgb_test

def xgb_model(x_train, y_train, x_test):
    xgb_train, xgb_test = cv_model(xgb, x_train, y_train, x_test, "xgb")
    return xgb_train, xgb_test

def cat_model(x_train, y_train, x_test):
    cat_train, cat_test = cv_model(CatBoostRegressor, x_train, y_train, x_test, "cat") 
    return cat_train, cat_test

lgb_train, lgb_test = lgb_model(x_train, y_train, x_test)

xgb_train, xgb_test = xgb_model(x_train, y_train, x_test) 

# cat_train, cat_test = cat_model(x_train, y_train, x_test)

rh_test = lgb_test*0.5 + xgb_test*0.5
testA['isDefault'] = rh_test
testA[['id','isDefault']].to_csv('output.csv', index=False)

