import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cab
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_recall_fscore_support,roc_curve,auc,roc_auc_score
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import gc
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

#from featexp import get_univariate_plots#用于特征筛选，需要先安装featexp
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif']=['Simhei']
plt.rcParams['axes.unicode_minus']=False
import json
import jieba
import fasttext


#--------------------0.数据的简单分析--------------------

# In[355]:

#读取csv文件
base_info=pd.read_csv('train/base_info.csv')
annual_report_info=pd.read_csv('train/annual_report_info.csv')
tax_info=pd.read_csv('train/tax_info.csv')
change_info=pd.read_csv('train/change_info.csv')
news_info=pd.read_csv('train/news_info.csv')
other_info=pd.read_csv('train/other_info.csv')
entprise_info=pd.read_csv('train/entprise_info.csv')#企业标注信息：0表示无非法集资风险，1表示有。{0: 13884, 1: 981}
entprise_evaluate=pd.read_csv('entprise_evaluate.csv')#验证集

#打印输出每个文件的信息量，以及这些信息中包含多少个不重复的id
print('base_info shape:',base_info.shape,'id unique:',len(base_info['id'].unique()))
print('annual_report_info shape:',annual_report_info.shape,'id unique:',len(annual_report_info['id'].unique()))
print('tax_info shape:',tax_info.shape,'id unique:',len(tax_info['id'].unique()))
print('change_info shape:',change_info.shape,'id unique:',len(change_info['id'].unique()))
print('news_info shape:',news_info.shape,'id unique:',len(news_info['id'].unique()))
print('other_info shape:',other_info.shape,'id unique:',len(other_info['id'].unique()))
print('entprise_info shape:',entprise_info.shape,'id unique:',len(entprise_info['id'].unique()))
print('entprise_evaluate shape:',entprise_evaluate.shape,'id unique:',len(entprise_evaluate['id'].unique()))


#--------------------1.特征构建--------------------
# ###  tfidi处理经营范围(opscope)特征

# In[356]:


# tfidif 处理经营范围的特征
#cn_stopwords.txt来源于 https://github.com/goto456/stopwords
def stopwordslist():
    stopwords = [line.strip() for line in open('cn_stopwords.txt',encoding='UTF-8').readlines()]
    return stopwords
# 创建一个停用词列表
stopwords = stopwordslist()
stopwords+=['、', '；', '，', '）','（']
#
train_df_scope=base_info.merge(entprise_info)[['id','opscope','label']]
test_df_scope=base_info[base_info['id'].isin(entprise_evaluate['id'].unique().tolist())]
test_df_scope=test_df_scope.reset_index(drop=True)[['id','opscope']]
str_label_0=''
str_label_1=''
for index,name,opscope,label in train_df_scope.itertuples():
    # 结巴分词
    seg_text = jieba.cut(opscope.replace("\t", " ").replace("\n", " "))
    outline = " ".join(seg_text)
    out_str=""
    for per in outline.split():
        if per not in stopwords: 
            out_str += per
            out_str+=" "
    if label==0:
        str_label_0+=out_str
    else:
        str_label_1+=out_str
corpus=[str_label_0,str_label_1]
vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
word=vectorizer.get_feature_names()#获取词袋模型中的所有词语总共7175个词语
weight=tfidf.toarray()#将(2, 7175)tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
# for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
#     #
#     for j in range(len(word)):
#         print(word[j],weight[i][j])
#下面将会根据tfidi算出来的权重将经营范围的文本特征转换为数值(利用weight[1,:]也即各个词语在第二类(违法类中所占据的权重之和))
illegal_word_weights={}
for i in range(len(word)):
    illegal_word_weights[word[i]]=weight[1][i]


tfidi_opscope=[]
for index,name,opscope in base_info[['id','opscope']].itertuples():
    # 
    seg_text = jieba.cut(opscope.replace("\t", " ").replace("\n", " "))
    outline = " ".join(seg_text)
    tfidi_frt=0
    for per in outline.split():
        if per in illegal_word_weights: 
            tfidi_frt+=illegal_word_weights[per]
    tfidi_opscope.append(tfidi_frt)
base_info['tfidif_opscope']=tfidi_opscope
print('对opscope提取tfidif特征完毕..........')


# ##  change_info、other_info，news_info，annual_report_info,tax表格的简单特征构建

# In[357]:


#change_info
change_info_clean=change_info.drop(['bgrq','bgq','bgh'],axis=1)
change_info_clean = change_info_clean.groupby('id',sort=False).agg('mean')
change_info_clean=pd.DataFrame(change_info_clean).reset_index()
#other_info
#空值大于0.5的列都删除掉
buf_group = other_info.groupby('id',sort=False).agg('mean')
other_info_clean=pd.DataFrame(buf_group).reset_index()
other_info_clean=other_info_clean.fillna(-1)
other_info_clean = other_info_clean.groupby('id',sort=False).agg('mean')
other_info_clean=pd.DataFrame(other_info_clean).reset_index()
#news_info
news_info_clean=news_info.drop(['public_date'],axis=1)
#对object类型进行编码
news_info_clean['positive_negtive']=news_info_clean['positive_negtive'].fillna("中立")
#
dic={}
cate=news_info_clean.positive_negtive.unique()
for i in range(len(cate)):
    dic[cate[i]]=i
#
news_info_clean['positive_negtive']=news_info_clean['positive_negtive'].map(dic)
news_info_clean = news_info_clean.groupby('id',sort=False).agg('mean')
news_info_clean=pd.DataFrame(news_info_clean).reset_index()
#处理annual_report_info的数据
#空值大于0.5的列都删除掉
annual_report_info_clean=annual_report_info.dropna(thresh=annual_report_info.shape[0]*0.5,how='all',axis=1)
#对object类型进行编码
annual_report_info_clean['BUSSTNAME']=annual_report_info_clean['BUSSTNAME'].fillna("无")
dic = {'无':-1,'开业':0, '歇业':1, '停业':2, '清算':3}
#
annual_report_info_clean['BUSSTNAME']=annual_report_info_clean['BUSSTNAME'].map(dic)
annual_report_info_clean = annual_report_info_clean.groupby('id',sort=False).agg('mean')
annual_report_info_clean=pd.DataFrame(annual_report_info_clean).reset_index()
#处理tax数据
tax_info_clean=tax_info.copy()
tax_info_clean['START_DATE']=pd.to_datetime(tax_info_clean['START_DATE'])
tax_info_clean['END_DATE']=pd.to_datetime(tax_info_clean['END_DATE'])
tax_info_clean['gap_day']=(tax_info_clean['END_DATE']-tax_info_clean['START_DATE']).dt.total_seconds()//3600//24
tax_info_clean=tax_info_clean.drop(['START_DATE','END_DATE'],axis=1)
tax_info_clean['TAX_CATEGORIES']=tax_info_clean['TAX_CATEGORIES'].fillna("无")#17 unique
tax_info_clean['TAX_ITEMS']=tax_info_clean['TAX_ITEMS'].fillna("无")#275 TAX_ITEMS
#对object类型进行编码
dic={}
cate=tax_info_clean.TAX_CATEGORIES.unique()
for i in range(len(cate)):
    dic[cate[i]]=i
tax_info_clean['TAX_CATEGORIES']=tax_info_clean['TAX_CATEGORIES'].map(dic)
#
dic={}
cate=tax_info_clean.TAX_ITEMS.unique()
for i in range(len(cate)):
    dic[cate[i]]=i
tax_info_clean['TAX_ITEMS']=tax_info_clean['TAX_ITEMS'].map(dic)
tax_info_clean['income']=tax_info_clean['TAX_AMOUNT']/tax_info_clean['TAX_RATE']
#
tax_info_clean = tax_info_clean.groupby('id',sort=False).agg('mean')
tax_info_clean=pd.DataFrame(tax_info_clean).reset_index()
#税额分箱
tax_info_clean['TAX_AMOUNT']=tax_info_clean['TAX_AMOUNT'].fillna(tax_info_clean['TAX_AMOUNT'].median())
tax_info_clean['bucket_TAX_AMOUNT']=pd.qcut(tax_info_clean['TAX_AMOUNT'], 10, labels=False,duplicates='drop')
print('finished .............')


# ## base_info数据较为重要，需要构建诸多交叉特征以及特征分箱

# In[358]:


# #处理base_info数据
base_info['opto']=pd.to_datetime(base_info['opto']).fillna(pd.to_datetime(base_info['opto']).max())
base_info['opfrom']=pd.to_datetime(base_info['opfrom'])
base_info['gap_year']=(base_info['opto']-base_info['opfrom']).dt.total_seconds()//3600//24//365
base_info_clean=base_info.drop(['opscope','opfrom','opto'],axis=1)

#............................对object类型进行编码...............................
base_info_clean['industryphy']=base_info_clean['industryphy'].fillna("无")
base_info_clean['dom']=base_info_clean['dom'].fillna("无")
base_info_clean['opform']=base_info_clean['opform'].fillna("无")
base_info_clean['oploc']=base_info_clean['oploc'].fillna("无")
#
dic={}
cate=base_info_clean.industryphy.unique()
for i in range(len(cate)):
    dic[cate[i]]=i
base_info_clean['industryphy']=base_info_clean['industryphy'].map(dic)
#
dic={}
cate=base_info_clean.dom.unique()
for i in range(len(cate)):
    dic[cate[i]]=i
base_info_clean['dom']=base_info_clean['dom'].map(dic)
#
dic={}
cate=base_info_clean.opform.unique()
for i in range(len(cate)):
    dic[cate[i]]=i
base_info_clean['opform']=base_info_clean['opform'].map(dic)
#
dic={}
cate=base_info_clean.oploc.unique()
for i in range(len(cate)):
    dic[cate[i]]=i
base_info_clean['oploc']=base_info_clean['oploc'].map(dic)
#
base_info_clean=base_info_clean.fillna(-1)
#
print('编码完毕.................')
#........................分箱.................................
def bucket(name,bucket_len):
    gap_list=[base_info_clean[name].quantile(i/bucket_len) for i in range(bucket_len+1)]#以分位数作为分箱标志
    # len_data=len(base_info_clean[name])
    new_col=[]
    for i in base_info_clean[name].values:
        for j in range(len(gap_list)):
            if gap_list[j]>=i:
                encode=j
                break
        new_col.append(encode)
    return new_col
#注册资本_实缴资本
base_info_clean['regcap_reccap']=base_info_clean['regcap']-base_info_clean['reccap']
#注册资本分箱
base_info_clean['regcap']=base_info_clean['regcap'].fillna(base_info_clean['regcap'].median())
base_info_clean['bucket_regcap']=pd.qcut(base_info_clean['regcap'], 10, labels=False,duplicates='drop')
#实缴资本分箱
base_info_clean['reccap']=base_info_clean['reccap'].fillna(base_info_clean['reccap'].median())
base_info_clean['bucket_reccap']=pd.qcut(base_info_clean['reccap'], 10, labels=False,duplicates='drop')
#注册资本_实缴资本分箱
base_info_clean['regcap_reccap']=base_info_clean['regcap_reccap'].fillna(base_info_clean['regcap_reccap'].median())
base_info_clean['bucket_regcap_reccap']=pd.qcut(base_info_clean['regcap_reccap'], 10, labels=False,duplicates='drop')
#.............................交叉.........................
#作两个特征的交叉
def cross_two(name_1,name_2):
    new_col=[]
    encode=0
    dic={}
    val_1=base_info_clean[name_1]
    val_2=base_info_clean[name_2]
    for i in tqdm(range(len(val_1))):
        tmp=str(val_1[i])+'_'+str(val_2[i])
        if tmp in dic:
            new_col.append(dic[tmp])
        else:
            dic[tmp]=encode
            new_col.append(encode)
            encode+=1
    return new_col
#作企业类型-小类的交叉特征
base_info_clean['enttypegb']=base_info_clean['enttypegb'].fillna("无")
base_info_clean['enttypeitem']=base_info_clean['enttypeitem'].fillna("无")
new_col=cross_two('enttypegb','enttypeitem')#作企业类型-小类的交叉特征
base_info_clean['enttypegb_enttypeitem']=new_col
#
#行业类别-细类的交叉特征
base_info_clean['industryphy']=base_info_clean['industryphy'].fillna("无")
base_info_clean['industryco']=base_info_clean['industryco'].fillna("无")
new_col=cross_two('industryphy','industryco')#作企业类型-小类的交叉特征
base_info_clean['industryphy_industryco']=new_col
#企业类型-行业类别的交叉特征
new_col=cross_two('enttypegb','industryphy')#作企业类型-小类的交叉特征
base_info_clean['enttypegb_industryphy']=new_col
#行业类别-企业类型小类的交叉特征
new_col=cross_two('industryphy','enttypeitem')#作企业类型-小类的交叉特征
base_info_clean['industryphy_enttypeitem']=new_col
#行业类别细类--企业类型小类的交叉特征
new_col=cross_two('industryco','enttypeitem')#作企业类型-小类的交叉特征
base_info_clean['industryco_enttypeitem']=new_col

#企业类型-小类-行业类别-细类的交叉特征
new_col=cross_two('enttypegb_enttypeitem','industryphy_industryco')#作企业类型-小类的交叉特征
base_info_clean['enttypegb_enttypeitem_industryphy_industryco']=new_col
base_info_clean.shape


# ## category特征单独提取出来

# In[359]:


cat_features=['industryphy','dom','opform','oploc','bucket_regcap',
              'bucket_reccap','bucket_regcap_reccap',
              'enttypegb','enttypeitem','enttypegb_enttypeitem',
              'enttypegb_industryphy','enttypegb_enttypeitem_industryphy_industryco',
              'industryphy','industryco','industryphy_industryco',
              'industryphy_enttypeitem','industryco_enttypeitem',
              'adbusign','townsign','regtype','TAX_CATEGORIES','bucket_TAX_AMOUNT',
              'legal_judgment_num','brand_num','patent_num','positive_negtive'
             ]


# In[360]:


#
all_data=base_info_clean.merge(annual_report_info_clean,how='outer')
all_data=all_data.merge(tax_info_clean,how='outer')
all_data=all_data.merge(change_info_clean,how='outer')
all_data=all_data.merge(news_info_clean,how='outer')
all_data=all_data.merge(other_info_clean,how='outer')
all_data=all_data.fillna(-1)
all_data[cat_features]=all_data[cat_features].astype(int)
all_data.shape#,base_info.shape,annual_report_info.shape,tax_info.shape


# In[361]:


#
train_df=all_data.merge(entprise_info)
train_data=train_df.drop(['id','label'],axis=1)
kind=train_df['label']
test_df=all_data[all_data['id'].isin(entprise_evaluate['id'].unique().tolist())]
test_df=test_df.reset_index(drop=True)
test_data=test_df.drop(['id'],axis=1)
train_data.shape,test_data.shape




# In[364]:


def eval_score(y_test,y_pre):
    _,_,f_class,_=precision_recall_fscore_support(y_true=y_test,y_pred=y_pre,labels=[0,1],average=None)
    fper_class={'合法':f_class[0],'违法':f_class[1],'f1':f1_score(y_test,y_pre)}
    return fper_class
#
def k_fold_serachParmaters(model,train_val_data,train_val_kind):
    mean_f1=0
    mean_f1Train=0
    n_splits=5
    sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
    for train, test in sk.split(train_val_data, train_val_kind):
        x_train = train_val_data.iloc[train]
        y_train = train_val_kind.iloc[train]
        x_test = train_val_data.iloc[test]
        y_test = train_val_kind.iloc[test]

        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        fper_class =  eval_score(y_test,pred)
        mean_f1+=fper_class['f1']/n_splits
        #print(fper_class)
        
        pred_Train = model.predict(x_train)
        fper_class_train =  eval_score(y_train,pred_Train)
        mean_f1Train+=fper_class_train['f1']/n_splits
    #print('mean valf1:',mean_f1)
    #print('mean trainf1:',mean_f1Train)
    return mean_f1


# In[365]:


xlf=xgb.XGBClassifier(max_depth=8 #7
                      ,learning_rate=0.1 #0.05
                      ,n_estimators=55
                      ,reg_alpha=0.005
                      ,n_jobs=8
                      ,importance_type='total_cover'
                     )
#
llf=lgb.LGBMClassifier(num_leaves=9
                           ,max_depth=5
                           ,learning_rate=0.05 
                           ,n_estimators=80
                           ,n_jobs=8
                           )
  
clf=cab.CatBoostClassifier(iterations=80 #60
                              ,learning_rate=0.05 #0.05
                              ,depth=10
                              ,silent=True
                              ,thread_count=8
                              ,task_type='CPU'
                              ,cat_features=cat_features
                              )

rf = (oob_score=True, random_state=2020,
            n_estimators= 100,max_depth=13,min_samples_split=5) #70, 13, 5
k_fold_serachParmaters(rf,train_data,kind)
print('xlf:',k_fold_serachParmaters(xlf,train_data,kind))
print('llf:',k_fold_serachParmaters(llf,train_data,kind))
print('clf:',k_fold_serachParmaters(clf,train_data,kind)) 
print('rf:',k_fold_serachParmaters(rf,train_data,kind)) 


# xlf: 0.831114697857251
# llf: 0.836176500865795
# clf: 0.8418842032860574
# rf: 0.8307529250979306
# 每1次验证的f1:0.8534031413612565
# 每2次验证的f1:0.8658227848101265
# 每3次验证的f1:0.8452088452088452
# 每4次验证的f1:0.8585607940446649
# 每5次验证的f1:0.8542713567839197
# mean f1: 0.8554533844417627

# In[366]:


# #
details = []
answers = []
mean_f1=0
n_splits=5
sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
cnt=0
for train, test in sk.split(train_data, kind):
    x_train = train_data.iloc[train]
    y_train = kind.iloc[train]
    x_test = train_data.iloc[test]
    y_test = kind.iloc[test]

    xlf.fit(x_train, y_train)
    pred_xgb = xlf.predict(x_test)
    weight_xgb = eval_score(y_test,pred_xgb)['f1']

    llf.fit(x_train, y_train)
    pred_llf = llf.predict(x_test)
    weight_lgb = eval_score(y_test,pred_llf)['f1']

    clf.fit(x_train, y_train)
    pred_cab = clf.predict(x_test)
    weight_cab =  eval_score(y_test,pred_cab)['f1']

    rf.fit(x_train, y_train)
    pred_rf = rf.predict(x_test)
    weight_rf =  eval_score(y_test,pred_rf)['f1']


    prob_xgb = xlf.predict_proba(x_test)
    prob_lgb = llf.predict_proba(x_test)
    prob_cab = clf.predict_proba(x_test)
    prob_rf = rf.predict_proba(x_test)

    scores = []
    ijkl = []
    weight = np.arange(0, 1.05, 0.1)
    for i, item1 in enumerate(weight):
        for j, item2 in enumerate(weight[weight <= (1 - item1)]):
            for k, item3 in enumerate(weight[weight <= (1 - item1-item2)]):
                prob_end = prob_xgb * item1 + prob_lgb * item2 + prob_cab *item3+prob_rf*(1 - item1 - item2-item3)
                #prob_end = np.sqrt(prob_xgb**2 * item1 + prob_lgb**2 * item2 + prob_cab**2 *item3+prob_rf**2*(1 - item1 - item2-item3))
                score = eval_score(y_test,np.argmax(prob_end,axis=1))['f1']
                scores.append(score)
                ijkl.append((item1, item2,item3, 1 - item1 - item2-item3))

    ii = ijkl[np.argmax(scores)][0]
    jj = ijkl[np.argmax(scores)][1]
    kk = ijkl[np.argmax(scores)][2]
    ll = ijkl[np.argmax(scores)][3]

    details.append(max(scores))
    details.append(weight_xgb)
    details.append(weight_lgb)
    details.append(weight_cab)
    details.append(weight_rf)
    details.append(ii)
    details.append(jj)
    details.append(kk)
    details.append(ll)

    cnt+=1
    print('每{}次验证的f1:{}'.format(cnt,max(scores)))
    mean_f1+=max(scores)/n_splits

    test_xgb = xlf.predict_proba(test_data)
    test_lgb = llf.predict_proba(test_data)
    test_cab = clf.predict_proba(test_data)
    test_rf = rf.predict_proba(test_data)
    #加权平均
    # ans = test_xgb * ii + test_lgb * jj + test_cab * kk + test_rf*ll#加权平均
    #加权平方平均
    ans = np.sqrt(test_xgb**2 * ii + test_lgb**2 * jj + test_cab**2 * kk + test_rf**2*ll)
    answers.append(ans)
print('mean f1:',mean_f1)


# In[367]:


df=pd.DataFrame(np.array(details).reshape(int(len(details)/9),9)
                ,columns=['test_end_score','xgboost','lightgbm','catboost','rf'
                ,'weight_xgboost','weight_lightgbm','weight_catboost','weight_rf'])
df


# In[368]:


df.mean()


# In[369]:


#
fina=sum(answers)/n_splits#
#fina=np.sqrt(sum(np.array(answers)**2)/n_splits)#平方平均
fina=fina[:,1]
test_df['score']=fina#可选:fina_persudo是伪标签的预测结果
submit_csv=test_df[['id','score']]
save_path='submit'+'.csv'
submit_csv.to_csv(save_path,index=False)
save_path
