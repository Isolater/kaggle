import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_path = 'input'
train = pd.read_csv('%s/%s' % (data_path, 'train.csv'))
test = pd.read_csv('%s/%s' % (data_path, 'test.csv'))
# print(train.info())
# 存活死亡人数
# print(train['Survived'].value_counts())
# 相关性协方差corr()函数
train_corr = train.drop('PassengerId', axis=1).corr()
# print(train_corr)
# 热力图
# f = plt.subplots(figsize=(15, 9))
# f = sns.heatmap(train_corr, vmin=-1, vmax=1, annot=True, square=True)

# 画图
# fig = plt.figure()
# ax = fig.add_subplot(111)

# Fare分布
# ax.hist(train['Fare'], bins=5)
# plt.title('Fare distribution')
# plt.xlabel('Fare')

# Ticket分布
# ax.hist(train['Pclass'], bins=5)
# plt.title('Pclass distribution')
# plt.xlabel('Pclass')
# plt.ylabel('Passenger')

# Survived和Pclass的相关性
# print(train.groupby(['Pclass'])['Pclass', 'Survived'].mean())
# train[['Pclass', 'Survived']].groupby(['Pclass']).mean().plot.bar()

# Survived和Sex的相关性
# print(train.groupby(['Sex'])['Sex', 'Survived'].mean())
# train.groupby(['Sex'])['Sex', 'Survived'].mean().plot.bar()

# SibSp,Parch和Survived的相关性
# print(train[['SibSp', 'Survived']].groupby(['SibSp']).mean())
# print(train[['Parch', 'Survived']].groupby(['Parch']).mean())
# train[['Parch', 'Survived']].groupby(['Parch']).mean().plot.bar()
# train[['SibSp', 'Survived']].groupby(['SibSp']).mean().plot.bar()
# Age和Survived的相关性
# a = sns.FacetGrid(train, hue='Survived', aspect=3)
# a.map(sns.distplot, 'Age')
# plt.show()

# 先将数据集合并，以便合理预测缺失值
test['Survived'] = 0
train_test = train.append(test)

# Pclass分列
train_test = pd.get_dummies(train_test, columns=['Pclass'])

# Sex分列
train_test = pd.get_dummies(train_test, columns=['Sex'])

# SlibSp and Parch
train_test['SibSp_Parch'] = train_test['SibSp'] + train_test['Parch']
train_test = pd.get_dummies(train_test, columns=['SibSp', 'Parch', 'SibSp_Parch'])

#Embarked
train_test = pd.get_dummies(train_test, columns=['Embarked'])

#Name
train_test['Name1'] = train_test['Name'].str.extract('.+,(.+)', expand=False).str.extract('^(.+?)\.', expand=False).str.strip()
# train_test['Name1'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer' , inplace = True)
# train_test['Name1'].replace(['Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty' , inplace = True)
# train_test['Name1'].replace(['Mme', 'Ms', 'Mrs'], 'Mrs')
# train_test['Name1'].replace(['Mlle', 'Miss'], 'Miss')
# train_test['Name1'].replace(['Mr'], 'Mr' , inplace = True)
# train_test['Name1'].replace(['Master'], 'Master' , inplace=True)
# train_test = pd.get_dummies(train_test,columns=['Name1'])
print(train_test['Name1'].value_counts())
#Fare
# train_test.loc[train_test['Fare'].isnull()]
#train.groupby(by=['Pclass', 'Embarked']).Fare.mean()
train_test['Fare'].fillna(14.644083, inplace=True)

#Ticket
# train_test['Ticket_Letter'] = train_test['Ticket'].str.split().str[0]
# train_test['Ticket_Letter'] = train_test['Ticket_Letter'].apply(lambda x: np.nan if x.isnumneric() else x)
# train_test.drop('Ticket', inplace=True, axis=1)
# train_test = pd.get_dummies(train_test, columns=['Ticket_Letter'], drop_first=True)

# Cabin
train_test.loc[train_test['Cabin'].isnull(), 'Cabin_nan'] = 1
train_test.loc[train_test['Cabin'].notnull(), 'Cabin_nan'] = 0
train_test = pd.get_dummies(train_test, columns=['Cabin_nan'])
train_test.drop('Cabin', axis=1, inplace=True)

# 划分数据集
train_data = train_test[:891]
test_data = train_test[891:]
train_data_X = train_data.drop(['Survived'], axis=1)
train_data_Y = train_data['Survived']
test_data_X = test_data.drop(['Survived'], axis=1)

# 数据规约
from sklearn.preprocessing import StandardScaler
ss2 = StandardScaler()
# ss2.fit(train_data_X)
# train_data_X_sd = ss2.transform(train_data_X)
# test_data_X_sd = ss2.transform(test_data_X)
# print(train_data_X)