import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from scipy import stats 

# 数据读取
train = pd.read_csv('train.csv')
testA = pd.read_csv('testA.csv')

# 数据预处理
# 【id】无意义，舍弃
# 【loanAmnt】无NAN值，数据共有8个“峰”，在保留绝大部分信息的情况下可以考虑将其分为8类:[0,40000] / 8
# 【term】无NAN值，取值范围{3，5}
# 【interestRate】无NAN值，数据呈偏态分布，使用log变换使其更接近于高斯分布
# 【installment】 无NAN值，护具呈偏态分布，使用sqrt变换使其接近于高斯分布
# 【grade】属性是【subGrade】的前缀，故只需保留后者即可
# 【employmentTitle】呈现出一种极为“偏激”的分布...还是直接扔进去训练吧
# 【employmentLength】 去掉“years”后，”< 1”转换成“0”，"10+"转换成"10"，将NAN值转换成加权平均数“6”
# 【homeOwnership】共有0-5六个值，其中0和1两个值的占比达到了89%以上，因此可以将值”>1“的归为同一类，最终得到三个值
# 【annualIncome】毫无规律，直接扔进去训练
# 【verificationStatus】该属性就只有三个值
# 【issueDate】将其中的年份与employmentLength作和(需要先完成其转换)
# 【isDefault】 需要得到的预测值
# 【purpose】一共就十三个值，直接扔进去训练吧
# 【postCode】 九百多个值，直接扔进去训练吧
# 【regionCode】五十个值，直接训练
# 【dti】数据范围从-1到999，粒度为0.01，但绝大部分数据都在0-40之间近似呈高斯分布，
#       其余范围只有零散的几个数据，因此可以将大于40的数据随机分布到40-41，将NAN值“归结到峰值”20，
#       再使用sqrt变化使得整个数据呈正态分布。
# 【delinquency_2years】数据范围0-39，但绝大部分数据都集中在0、1、2之间，因此可以将大于3的数据都归入到4（表示3次以上）中
# 【ficoRangeLow】与【ficoRangeHigh】只需要保留其中一个即可，因为两者的差中99.985%是4，0.015%是5，
#       而且其表示的含义是“信用分”。ficoRangeLow的数据分布是“一半高斯分布”，且数值较少，全部保留。
# 【openAcc】一共75个值，但是绝大部分在[0，35]内呈现出偏态分布的状况，因此将>35的数据随机分布在36-41之间
# 【pubRec】一共三十多个值，但是其中97.24%的值是0或1，因此可以将2以上的数值全部归为2（表示两次或以上）
# 【pubRecBankruptcies】一共11个值，而且有NAN值，单几乎绝大部分出现在0和1，因此将NAN值以及2以上的数值都归为2
# 【revolBal】直接丢进去训练
# 【revolUtil】其105内的值是呈类似于正态分布的，但是＞105的数据就很稀疏，可以将其与NAN值随机分布到105-110之间
# 【totalAcc】期75以上的数据比较稀疏，随机分不到75-85之间
# 【initialListStatus】只有0，1两种取值
# 【applicationType】 只有0，1两种取值
# 【earliesCreditLine】直接扔进去训练
# 【title】直接扔进去训练
# 【policyCode】全是1、无意义，舍弃
# 【n1】将NAN值分配到加权平均数3
# 【n2】将NAN值分配到加权平均数6
# 【n3】将NAN值分配到加权平均数6
# 【n3】将NAN值分配到加权平均数5
# 【n4】将NAN值分配到加权平均数5
# 【n5】将NAN值分配到加权平均数8
# 【n6】将NAN值分配到加权平均数9
# 【n7】将NAN值分配到加权平均数8
# 【n8】将NAN值分配到加权平均数15
# 【n9】将NAN值分配到加权平均数6
# 【n10】将NAN值分配到加权平均数12
# 【n11】0占91.2%，NAN占8.7%，1~4占0.2%
# 【n12】0占94.66%,NAN占5%,1-4占0.34%
# 【n13】0占89.5%，1占4%，NAN占5%，1-39占1.5%

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

def n13_func(x):
    if pd.isnull(x):
        return 0
    elif x > 1:
        return 12 #+np.random.randint(low = 1, high = 6, dtype = np.int64)
    else:
        return x
train['n13'] = train['n13'].apply(n13_func)

print(train['n14'].value_counts(dropna=False, normalize = False).sort_index())
train['n14'].plot(kind='hist', bins = 16)#, edgecolor="black")
plt.show()




# 【interestRate】
# train['interestRate'] = np.log1p(train['interestRate'])
# sns.distplot(train['interestRate'],fit=norm)
# (mu, sigma) = norm.fit(train['interestRate'])
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
# plt.ylabel('Frequency')
# plt.title('interestRate distribution after log transform')
# plt.show()


# # 【installment】
# train['installment'] = np.sqrt(train['installment'])
# sns.distplot(train['installment'],fit=norm)
# (mu, sigma) = norm.fit(train['installment'])
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
# plt.ylabel('Frequency')
# plt.title('installment distribution after sqrt transform')
# plt.show()


# 【employmentLength】
# train['employmentLength'].replace(to_replace='10+ years', value='10 years', inplace=True)
# train['employmentLength'].replace('< 1 year', '0 years', inplace=True)
# # 用于提取employmentLength中的数字
# def employmentLength_to_int(s):
#     if pd.isnull(s):
#         return np.int8(6)
#     else:
#         return np.int8(s.split()[0])
# # DataFrame.apply() 函数则会遍历行或列的每一个元素，对元素运行指定的 function。
# train['employmentLength'] = train['employmentLength'].apply(employmentLength_to_int)


#【issueDate】
# def issueDate_func(x):
#     return np.int64(x.split('-')[0])


# train['issueDate'] = train['issueDate'].apply(issueDate_func) + train['employmentLength']


# 【dti】
# def dti_func(x):
#     if pd.isnull(x):
#         return 20
#     elif x == -1:
#         return 0
#     elif x > 40:
#         k = 40+np.random.randint(low = 0, high = 200, dtype = np.int64)*0.01
#         print(k)
#         return k
#     else:
#         return x
# train['dti'] = train['dti'].apply(dti_func)
# train['dti'] = np.log1p(train['dti'])
# print(train['dti'].value_counts(dropna=False, normalize = False).sort_index())
# train['dti'].plot(kind='hist', bins = 420)#, edgecolor="black")
# plt.show()
# sns.distplot(train['dti'],fit=norm)
# (mu, sigma) = norm.fit(train['dti'])
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
# plt.ylabel('Frequency')
# plt.title('dti distribution after sqrt transform')
# plt.show()


# 【delinquency_2years】
# def delinquency_func(x):
#     if x > 3:
#         return 4
#     else:
#         return x        
# train['delinquency_2years'] = train['delinquency_2years'].apply(delinquency_func)


#【pubRec】
# train['pubRec'] = train['pubRec'].apply(lambda x: 2 if x>2 else x)


#【revolUtil】
# train['revolUtil'] = train['revolUtil'].apply(lambda x: 105 + 0.1 * np.random.randint(low = 0, high = 51, dtype = np.int64) if (x>105 or pd.isnull(x)) else x)


#【totalAcc】
# train['totalAcc'] = train['totalAcc'].apply(lambda x: 75 +np.random.randint(low = 0, high = 11, dtype = np.int64) if x>75 else x)


#【n1】
# def n1_func(x):
#     if pd.isnull(x):
#         return 3
#     elif x > 15:
#         return 16 #+np.random.randint(low = 1, high = 6, dtype = np.int64)
#     else:
#         return x
# train['n1'] = train['n1'].apply(n1_func)

#【n2】
# def n2_func(x):
#     if pd.isnull(x):
#         return 6
#     elif x > 20:
#         return 21 #+np.random.randint(low = 1, high = 6, dtype = np.int64)
#     else:
#         return x
# train['n2'] = train['n2'].apply(n2_func)

#【n3】
# def n3_func(x):
#     if pd.isnull(x):
#         return 6
#     elif x > 20:
#         return 21 #+np.random.randint(low = 1, high = 6, dtype = np.int64)
#     else:
#         return x
# train['n3'] = train['n3'].apply(n3_func)

#【n4】
# def n4_func(x):
#     if pd.isnull(x):
#         return 5
#     elif x > 20:
#         return 21 #+np.random.randint(low = 1, high = 6, dtype = np.int64)
#     else:
#         return x
# train['n4'] = train['n4'].apply(n4_func)


#【n5】
# def n5_func(x):
#     if pd.isnull(x):
#         return 8
#     elif x > 30:
#         return 31 #+np.random.randint(low = 1, high = 6, dtype = np.int64)
#     else:
#         return x
# train['n5'] = train['n5'].apply(n5_func)


#【n6】
# def n6_func(x):
#     if pd.isnull(x):
#         return 9
#     elif x > 50:
#         return 51 #+np.random.randint(low = 1, high = 6, dtype = np.int64)
#     else:
#         return x
# train['n6'] = train['n6'].apply(n6_func)


#【n7】
# def n7_func(x):
#     if pd.isnull(x):
#         return 8
#     elif x > 30:
#         return 31 #+np.random.randint(low = 1, high = 6, dtype = np.int64)
#     else:
#         return x
# train['n7'] = train['n7'].apply(n7_func)

#【n8】
# def n8_func(x):
#     if pd.isnull(x):
#         return 15
#     elif x > 55:
#         return 56 #+np.random.randint(low = 1, high = 6, dtype = np.int64)
#     else:
#         return x
# train['n8'] = train['n8'].apply(n8_func)

#【n9】
# def n9_func(x):
#     if pd.isnull(x):
#         return 6
#     elif x > 25:
#         return 26 #+np.random.randint(low = 1, high = 6, dtype = np.int64)
#     else:
#         return x
# train['n9'] = train['n9'].apply(n9_func)

#【n10】
# def n10_func(x):
#     if pd.isnull(x):
#         return 12
#     elif x > 40:
#         return 41 #+np.random.randint(low = 1, high = 6, dtype = np.int64)
#     else:
#         return x
# train['n10'] = train['n10'].apply(n10_func)

#【n11】
# def n11_func(x):
#     if pd.isnull(x):
#         return 0
#     elif x > 1:
#         return 1 #+np.random.randint(low = 1, high = 6, dtype = np.int64)
#     else:
#         return x
# train['n11'] = train['n11'].apply(n11_func)

#【n12】
# def n12_func(x):
#     if pd.isnull(x):
#         return 0
#     elif x > 1:
#         return 1 #+np.random.randint(low = 1, high = 6, dtype = np.int64)
#     else:
#         return x
# train['n12'] = train['n12'].apply(n12_func)


#【n13】
# def n13_func(x):
#     if pd.isnull(x):
#         return 0
#     elif x > 1:
#         return 12 #+np.random.randint(low = 1, high = 6, dtype = np.int64)
#     else:
#         return x
# train['n13'] = train['n13'].apply(n13_func)

#【n14】
# def n14_func(x):
#     if pd.isnull(x):
#         return 2
#     elif x > 15:
#         return 15 #+np.random.randint(low = 1, high = 6, dtype = np.int64)
#     else:
#         return x
# train['n14'] = train['n14'].apply(n14_func)