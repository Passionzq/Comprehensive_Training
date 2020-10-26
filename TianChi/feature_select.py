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
from scipy.stats import norm
import seaborn as sns
from scipy import stats 
import random

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings('ignore')

# #显示所有列
# pd.set_option('display.max_columns', None)
# #显示所有行
# pd.set_option('display.max_rows', None)

#--------------------------数据读取--------------------------
train = pd.read_csv('train.csv')
testA = pd.read_csv('testA.csv')

# 调用read_csv()读取文件时时会自动识别表头（默认读取第一行，即header=0)，数据有表头时不能设置header为空；
train.head()


# --------------------------数据预处理--------------------------
# 将testA拼接到train的末端
data = pd.concat([train, testA], axis=0, ignore_index=True) 

# [1]id：无意义，舍弃

# [2]loanAmnt：无NAN值，数据共有8个“峰”，采用分箱法将其分为八份：[0,40000] / 8
data['loanAmnt'] = data['loanAmnt'].apply(lambda x: x//5000)

# [3]term：无NAN值，取值范围{3, 5}，不作处理

# [4]interestRate：无NAN值，数据呈偏态分布，使用log变换使其更接近于高斯分布
data['interestRate'] = np.log1p(data['interestRate'])

# [5]installment：无NAN值，数据呈偏态分布，使用sqrt变换使其接近于高斯分布
data['installment'] = np.sqrt(data['installment'])

# [6,7]grade属性是subGrade的前缀，故删除前者保留后者
def subGrade_func(x):
    return (ord(x[0]) - ord('A') + 1) * int(x[1])
data['subGrade'] = data['subGrade'].apply(subGrade_func)

# [8]employmentTitle：呈现出一种极为“偏激”的分布，不作处理
data['employmentTitle'] = data['employmentTitle'].apply(lambda x: 1 if pd.isnull(x) else x)

# [9]employmentLength：去掉“years”后，”< 1”转换成“0”，"10+"转换成"10"，将NAN值转换成加权平均数“6”
def employmentLength_to_int(s): 
    if pd.isnull(s):
        return np.int8(6)
    else:
        return np.int8(s.split()[0])

data['employmentLength'].replace(to_replace='10+ years', value='10 years', inplace=True)
data['employmentLength'].replace('< 1 year', '0 years', inplace=True)
data['employmentLength'] = data['employmentLength'].apply(employmentLength_to_int)

# [10]homeOwnership:一共只有0-5六个值选项，不作处理

# [11]annualIncome：无法看出相关规律，不作处理

# [12]verificationStatus：一共只有三个值选项，不作处理

# [13]issueDate：提取其中的年份
def issueDate_func(x):
    return np.int64(x.split('-')[0])

data['issueDate'] = data['issueDate'].apply(issueDate_func)

# [14]isDefault：需要求的预测值

# [15]purpose：一共只有13个值选项，不作处理

# [16]postCode：无法看出相关规律，不作处理

# [17]regionCode：一共就50个值选项，不作处理

# [18]dti:数据范围从-1到999，粒度为0.01，但绝大部分数据都在0-40之间近似呈高斯分布，
#         其余范围只有零散的几个数据，因此可以将大于40的数据随机分布到40-41，
#         将NAN值“归结到峰值”20，再使用sqrt变化使得整个数据呈正态分布。
def dti_func(x):
    if pd.isnull(x):
        return 20
    elif x == -1:
        return 0
    elif x > 40:
        return 40+np.random.randint(low = 0, high = 200, dtype = np.int64)*0.01
    else:
        return x

data['dti'] = np.sqrt(data['dti'].apply(dti_func))

# [19]delinquency_2years：数据范围0-39，但绝大部分数据都集中在0、1、2之间
#                         可以考虑将大于3的数据都归入到4（表示3次以上）中
# data['delinquency_2years'] = data['delinquency_2years'].apply(lambda x: 4 if x > 3 else x)

# [20,21]ficoRangeLow与ficoRangeHigh只需要保留其中一个即可，因为两者的差中99.985%是4，0.015%是5，
#        而且其表示的含义是“信用分”。ficoRangeLow的数据分布是“一半高斯分布”，且值选项较少，全部保留。

# [22]openAcc：一共75个值选项，但是绝大部分在[0，35]内呈现出偏态分布的状况，不作处理

# [23]pubRec:一共30+值选项，其中97.24%的值是0或1，可以考虑将其2以上的值都归到2中

# [24]pubRecBankruptcies：存在NAN值，将其分配到加权平均数0
data['pubRecBankruptcies'] = np.sqrt(data['pubRecBankruptcies'].apply(lambda x: 0 if pd.isnull(x) else x))

# [25]revolBal：没有发现规律，不作处理

# [26]revolUtil：其105内的值是呈类似于正态分布的，但是＞105的数据就很稀疏，还是不作处理了

# [27]totalAcc：没有发现规律，不作处理

# [28,29]initialListStatus和applicationType均只有0、1两种取值

# [30]earliesCreditLine：将其中的year转化成int型代替原来的'month-year'
data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda s: int(s[-4:]))

# [31]title：没有发现规律，不作处理
data['title'] = data['title'].apply(lambda x: 0 if pd.isnull(x) else x)

# [32]policyCode：全是1、无意义，舍弃

# [33]n0：将NAN值分配到加权平均数0.5（随机数决定0或1）
data['n0'] = data['n0'].apply(lambda x: random.randint(0,1) if pd.isnull(x) else x)

# [33]n1：将NAN值分配到加权平均数3
data['n1'] = data['n1'].apply(lambda x: 3 if pd.isnull(x) else x)

# [34]n2：将NAN值分配到加权平均数6
data['n2'] = data['n2'].apply(lambda x: 6 if pd.isnull(x) else x)

# [35]n3：将NAN值分配到加权平均数6
data['n3'] = data['n3'].apply(lambda x: 6 if pd.isnull(x) else x)

# [36]n4：将NAN值分配到加权平均数5
data['n4'] = data['n4'].apply(lambda x: 5 if pd.isnull(x) else x)

# [37]n5：将NAN值分配到加权平均数8
data['n5'] = data['n5'].apply(lambda x: 8 if pd.isnull(x) else x)

# [38]n6：将NAN值分配到加权平均数9
data['n6'] = data['n6'].apply(lambda x: 9 if pd.isnull(x) else x)

# [39]n7：将NAN值分配到加权平均数8
data['n7'] = data['n7'].apply(lambda x: 8 if pd.isnull(x) else x)

# [40]n8：将NAN值分配到加权平均数15
data['n8'] = data['n8'].apply(lambda x: 15 if pd.isnull(x) else x)

# [41]n9：将NAN值分配到加权平均数6
data['n9'] = data['n9'].apply(lambda x: 6 if pd.isnull(x) else x)

# [42]n10：将NAN值分配到加权平均数12
data['n10'] = data['n10'].apply(lambda x: 12 if pd.isnull(x) else x)

# [43]n11：将NAN值分配到加权平均数0
data['n11'] = data['n11'].apply(lambda x: 0 if pd.isnull(x) else x)

# [44]n12：将NAN值分配到加权平均数0
data['n12'] = data['n12'].apply(lambda x: 0 if pd.isnull(x) else x)

# [45]n13：将NAN值分配到加权平均数0
data['n13'] = data['n13'].apply(lambda x: 0 if pd.isnull(x) else x)

# [46]n14：将NAN值分配到加权平均数2
data['n14'] = data['n14'].apply(lambda x: 2 if pd.isnull(x) else x)

# 部分类别特征
cate_features = ['subGrade', 'employmentTitle', 'homeOwnership', 'verificationStatus', 'purpose', 'postCode', 'regionCode', \
                 'applicationType', 'initialListStatus', 'title']

# （可删除）nunique():用于获取唯一值的统计次数。
# for f in cate_features:
#     print(f, '类型数：', data[f].nunique())

# 类型数在2之上，又不是高维稀疏的，需要使用get_dummies实现one hot encode
# 也即：就是添加原来数据中没有的变量，但是这并不是意味着可以随意添加，应该是根据原来的数据进行转换。
# 例如：将一个变量Embarked，根据它的值（C、Q、S）转换为Embarked_C、Embarked_Q、Embarked_S三个变量
# (转化后有默认名，也可以利用参数prefix来自己修改）
# data = pd.get_dummies(data, columns=['subGrade', 'homeOwnership', 'verificationStatus'], drop_first=True)

# 高维类别特征需要进行转换
# for f in ['employmentTitle', 'postCode', 'title']:
#     # 这个操作的目的是将数据按照${f}进行分组（因为将train表和testA表合并了）
#     # 然后统计每组的个数（count），并且新增一列'f_cnts'的属性，其值为相应的count
#     # tips：此操作并未将该列属性增加到csv表格中，如果需要则应使用set_index()
#     data[f+'_cnts'] = data.groupby([f])['id'].transform('count')

#     # 增加一组属性：对属性${f}进行降序（ascending = False）排，
#     # 并且使用astype()将其数值强制转化成int
#     data[f+'_rank'] = data.groupby([f])['id'].rank(ascending=False).astype(int)
    
#     # 删除${f}这列属性
#     del data[f]


features = [f for f in data.columns if f not in [
                                                'id','grade','ficoRangeHigh','isDefault','policyCode',\
                                                'postCode'
                                                ]]
features.remove('interestRate')
features.remove('installment')
features.remove('dti')
features.remove('pubRecBankruptcies')
features.remove('revolUtil')





# for i in features:
#     print(data[i].value_counts(dropna=False, normalize = False).sort_index())
#     print('----------------------------------------')

# print(data['n0'].value_counts(dropna=True, normalize = True).sort_index())

# 训练组为属性isDefault有明确值的数据，测试组则与之相反
train = data[data.isDefault.notnull()].reset_index(drop=True)
test = data[data.isDefault.isnull()].reset_index(drop=True)

x_train = train[features]
x_test = test[features]
y_train = train['isDefault']

names = list(data)
names.remove('id')
names.remove('grade')
names.remove('ficoRangeHigh')
names.remove('isDefault')
names.remove('policyCode')
names.remove('postCode')
names.remove('interestRate')
names.remove('installment')
names.remove('dti')
names.remove('pubRecBankruptcies')
names.remove('revolUtil')

rf = RandomForestRegressor()
rf.fit(x_train,y_train)
print(sorted(zip(map(lambda x: round(x,4),rf.feature_importances_),names),reverse=True))