import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from sklearn.feature_selection import f_regression, RFE
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor


data_train = pd.read_csv('input/train.csv')
data_test = pd.read_csv('input/test.csv')
data1 = data_train.copy(deep=True)
data_cleaner = [data1, data_test]

# print(data_train.head())
# print(data1.test.head())
# print(data_train.info())
# print(data1.isnull().sum())


train_ID = data_train['Id']
test_ID = data_test['Id']

# data_train.drop('Id', axis=1, inplace=True)
# data_test.drop('Id', axis=1, inplace=True)

# 皮尔逊相关系数
# train_corr = data_train.corr()
# k = 10
# cols = train_corr.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(data_train[cols].values.T)

# 热力图
# sns.set(font_scale=1.5)
# hm = plt.subplots(figsize=(20, 12))
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':10}, yticklabels=cols.values, xticklabels=cols.values)

#散点图，选择OverallQual,GrLivArea,GarageCars,TotalBssmtSF,FullBath,YearBuilt特征
# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF','FullBath', 'YearBuilt']
# sns.pairplot(data_train[cols], size=2.5)
# plt.show()

#RFE
# lr = LinearRegression()
# rfe = RFE(lr, n_features_to_select=1)
# Y = data_train['SalePrice']
# X = data_train.drop('SalePrice', axis=1)
# rfe.fit_transform(X, Y)
# print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_))))

data_test['SalePrice'] = None
train_test = pd.concat((data_train, data_test)).reset_index(drop=True)

total = train_test.isnull().sum().sort_values(ascending=False)
percent = (train_test.isnull().sum()/train_test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Lost Percent'])
# print(missing_data[missing_data.isnull().values==False].sort_values('Total', axis=0,ascending=False).head(20))
# train_test = train_test.drop((missing_data[missing_data['Total'] > 1]).index.drop('SalePrice'), axis=1)
# tmp = train_test[train_test['SalePrice'].isnull().values==False]
# print(tmp.isnull().sum().max())

# 去偏度
# fig = plt.figure(figsize=(12, 6))
# ax1 = fig.add_subplot(1, 2, 1)
# ax2 = fig.add_subplot(1, 2, 2)
# ax1.hist(data_train.SalePrice)
# ax2.hist(np.log1p(data_train.SalePrice))
# plt.show()
# print('Skewness: %f' % data_train['SalePrice'].skew())
# print('Kurtosis: %f' % data_train['SalePrice'].kurt())

# GrLivArea with SalePrice
# 删除异常值
# print(data_train.sort_values(by='GrLivArea', ascending=False)[:2])
# tmp = train_test[train_test['SalePrice'].isnull().values == False]
# train_test = train_test.drop(tmp[tmp['Id'] == 1299].index)
# train_test = train_test.drop(tmp[tmp['Id'] == 524].index)
# data_train = data_train.drop(data_train[(data_train['GrLivArea'] > 4000) & (data_train['SalePrice'] < 300000)].index)
var = 'GrLivArea'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.show()

#TotalBsmtSF with SalePrice
# var = 'TotalBsmtSF'
# data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
# data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# plt.show()

# 正太
train_test['SalePrice'] = [i if i is None else np.log1p(i) for i in train_test['SalePrice']]
# tmp = train_test[train_test['SalePrice'].isnull().values==False]
# sns.distplot(tmp[tmp['SalePrice'] != 0]['SalePrice'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(tmp['SalePrice'], plot=plt)
# plt.show()

# GrLivArea
train_test['GrLivArea'] = [i if i is None else np.log1p(i) for i in train_test['GrLivArea']]
# tmp = train_test[train_test['SalePrice'].isnull().values==False]
# sns.distplot(tmp['GrLivArea'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(tmp['GrLivArea'], plot=plt)
# plt.show()

# TotalBsmtSF
# 考虑到大量0值问题
data_train.loc[data_train['TotalBsmtSF'] == 0, 'TotalBsmtSF'] = 1
train_test['TotalBsmtSF'] = np.log1p(train_test['TotalBsmtSF'])
# tmp = train_test[train_test['SalePrice'].isnull().values == False]
# tmp = np.array(tmp.loc[tmp['TotalBsmtSF'] > 0, ['TotalBsmtSF']])[:, 0]
# sns.distplot(tmp, fit=norm)
# fig = plt.figure()
# res = stats.probplot(tmp, plot=plt)
# plt.show()


#同方差性
# SalePrice and GrLivArea
# tmp = train_test[train_test['SalePrice'].isnull().values==False]
# plt.scatter(tmp['GrLivArea'], tmp['SalePrice'])
# plt.show()

# SalePrice and TotalBsmtSF
# tmp = train_test[train_test['SalePrice'].isnull().values==False]
# plt.scatter(tmp[tmp['TotalBsmtSF'] > 0]['TotalBsmtSF'], tmp[tmp['TotalBsmtSF'] >0]['SalePrice'])
# plt.show()

#模型选择
#数据标准化
# tmp = train_test[train_test['SalePrice'].isnull().values==False]
# tmp_1 = train_test[train_test['SalePrice'].isnull().values==True]
#
# x_train = tmp[['OverallQual', 'GrLivArea','GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']]
# y_train = tmp[['SalePrice']].values.ravel()
# x_test = tmp_1[['OverallQual', 'GrLivArea','GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']]
#
# x_test['GarageCars'].fillna(x_test.GarageCars.median(), inplace=True)
# x_test['TotalBsmtSF'].fillna(x_test.TotalBsmtSF.median(), inplace=True)

#开始建模
# ridge = Ridge(alpha=0.1)

# bagging
# params = [1, 20, 40, 60]
# test_scores = []
# for param in params:
#     clf = BaggingRegressor(base_estimator=ridge, n_estimators=param)
#     test_score = np.sqrt(-cross_val_score(clf, x_train, y_train, cv=10, scoring='neg_mean_squared_error'))
#     test_scores.append(np.mean(test_score))
#
# print(test_score.mean())
# plt.plot(params, test_scores)
# plt.title('n_estimators vs CV Error')
# plt.show()

# train_sizes, train_loss, test_loss = learning_curve(
#     ridge, x_train, y_train, cv=10, scoring='neg_mean_squared_error',
#     train_sizes=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1]
# )

# 训练误差均值
# train_loss_mean = -np.mean(train_loss, axis=1)
# test_loss_mean = -np.mean(test_loss, axis=1)
#
# plt.plot(train_sizes/len(x_train), train_loss_mean, 'o-', color='r', label='Training')
# plt.plot(train_sizes/len(x_train), test_loss_mean, 'o-', color='g', label='Cross-Validation')
#
# plt.xlabel('Training data size')
# plt.ylabel('Loss')
# plt.legend(loc = 'best')
# plt.show()

# mode_br = BaggingRegressor(base_estimator=ridge, n_estimators=10)
# mode_br.fit(x_train, y_train)
# y_test = np.expm1(mode_br.predict(x_test))
#
# submission_df = pd.DataFrame(data = {'Id':data_test['Id'], 'SalePrice':y_test})
# submission_df.to_csv('input/submission.csv', columns=['Id', 'SalePrice'], index=False)