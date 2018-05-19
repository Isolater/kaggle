import sys
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix

import numpy as np
import scipy as sp

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

import random
import time

data_raw = pd.read_csv('input/train.csv')
data_val = pd.read_csv('input/test.csv')
data1 = data_raw.copy(deep=True)
data_cleaner = [data1, data_val]

# preview data
# print(data_raw.info())
# print(data1.isnull().sum())

for dataset in data_cleaner:
    # complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)

    # complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)

    # complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

drop_column = ['PassengerId', 'Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace=True)
# print(data1.isnull().sum())
# print(data_val.isnull().sum())

for dataset in data_cleaner:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1  # 1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0

    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

stat_min = 10

title_names = (data1['Title'].value_counts() < stat_min)
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
# print(data1['Title'].value_counts())

label = LabelEncoder()
for dataset in data_cleaner:
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])

Target = ['Survived']
data1_x = ['Sex', 'Pclass', 'Embarked', 'Title', 'SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']
data1_x_calc = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code', 'SibSp', 'Parch', 'Age', 'Fare']
data1_xy = Target + data1_x
# print('Original X Y:', data1_xy, '\n')

data1_x_bin = ['Sex_Code', 'Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
# print('Bin X Y:', data1_xy_bin, '\n')

data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
# print('Dummy X Y:', data1_xy_dummy, '\n')
#
# print(data1_dummy.head())
#
# print('Train columns with null values:\n', data1.isnull().sum())
# print('-'*10)
# print(data1.info())
# print('-'*10)
#
#
# print('Test/Validaton columns with null values: \n', data_val.isnull().sum())
# print('-'*10)
# print(data_val.info())
# print('-'*10)
#
# print(data_raw.describe(include='all'))

train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state=0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target], random_state=0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state=0)

# print('Data1 Shape: {}'.format(data1.shape))
# print('Train1 Shape:{}'.format(train1_x.shape))
# print('Test1 Shape:{}'.format(test1_x.shape))
# print(train1_x_bin.head())

# for x in data1_x:
#     if data1[x].dtype != 'float64':
#         print('Survival Correlation by:', x)
#         print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())
#         print('-'*10, '\n')
#
# print(pd.crosstab(data1['Title'], data1[Target[0]]))


# plt.figure(figsize=[16, 12])
#
# plt.subplot(231)
# plt.boxplot(x=data1['Fare'], showmeans=True, meanline=True)
# plt.title('Fare Boxplot')
# plt.ylabel('Fare ($)')
#
# plt.subplot(232)
# plt.boxplot(x=data1['Age'], showmeans=True, meanline=True)
# plt.title('Age Boxplot')
# plt.ylabel('Age (Years)')
#
# plt.subplot(233)
# plt.boxplot(x=data1['FamilySize'], showmeans=True, meanline=True)
# plt.title('Family Size Boxplot')
# plt.ylabel('Family Size (#)')
#
# plt.subplot(234)
# plt.hist(x=[data1[data1['Survived'] == 1]['Fare'], data1[data1['Survived'] == 0]['Fare']], stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])
# plt.title('Fare Histogram by Survival')
# plt.xlabel('Fare ($)')
# plt.ylabel('# of Passengers')
# plt.legend()
#
# plt.subplot(235)
# plt.hist(x=[data1[data1['Survived'] == 1]['Age'], data1[data1['Survived'] == 0]['Age']], stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])
# plt.title('Age Histogram by Survival')
# plt.xlabel('Age (Years)')
# plt.ylabel('# of Passengers')
# plt.legend()
#
# plt.subplot(236)
# plt.hist(x=[data1[data1['Survived'] == 1]['FamilySize'], data1[data1['Survived'] == 0]['FamilySize']], stacked=True, color=['g', 'r'], label=['Survived', 'Dead'])
# plt.title('Family Size Histogram by Survival')
# plt.xlabel('Family Size (#)')
# plt.ylabel('# of Passengers')
# plt.legend()
#
# fig, saxis = plt.subplots(2, 3, figsize=(16, 12))
# sns.barplot(x='Embarked', y='Survived', data=data1, ax=saxis[0, 0])
# sns.barplot(x='Pclass', y='Survived', order=[1, 2, 3], data=data1, ax=saxis[0, 1])
# sns.barplot(x='IsAlone', y='Survived', order=[1, 0], data=data1, ax=saxis[0, 2])
#
# sns.pointplot(x='FareBin', y='Survived', data=data1, ax=saxis[1, 0])
# sns.pointplot(x='AgeBin', y='Survived', data=data1, ax=saxis[1, 1])
# sns.pointplot(x='FamilySize', y='Survived', data=data1, ax=saxis[1, 2])
#
#
# fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(14, 12))
#
# sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=data1, ax=axis1)
# axis1.set_title('Pclass vs Fare Survival Comparison')
#
# sns.violinplot(x='Pclass', y='Age', hug='Survived', data=data1, split=True, ax=axis2)
# axis2.set_title('Pclass vs Age Survival Comparison')
#
# sns.boxplot(x='Pclass', y='FamilySize', hue='Survived', data=data1, ax=axis3)
# axis2.set_title('Pclass vs Family Size Survival Comparison')
#
# fig, qaxis = plt.subplots(1, 3, figsize=(14, 12))
#
# sns.barplot(x='Sex', y='Survived', hue='Embarked', data=data1, ax=qaxis[0])
# axis1.set_title('Sex vs Embarked Survival Comparison')
#
# sns.barplot(x='Sex', y='Survived', hue='Pclass', data=data1, ax=qaxis[1])
# axis1.set_title('Sex vs Pclass Survival Comparison')
#
# sns.barplot(x='Sex', y='Survived', hue='IsAlone', data=data1, ax=qaxis[2])
# axis1.set_title('Sex vs IsAlone Survival Comparison')
#
#
# fig, (maxis1, maxis2) = plt.subplots(1, 2, figsize(14, 12))
#
# sns.pointplot(x='FamilySize', y='Survived', hue='Sex', data=data1, palette={'male':'blue', 'female':'pink'}, markers=['*', '~'], linestyles=['-', '--'], ax=maxis1)
#
# sns.pointplot(x='Pclass', y='Survived', hue='Sex', data=data1, palette={'male':'blue', 'female':'pink'}, markers=['*','~'], linestyles=['-', '--'], ax=maxis2)
#
#
# e = sns.FacetGrid(data1, col='Embarked')
# e.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette='deep')
# e.add_legend()
#
#
# a = sns.FacetGrid(data1, hue='Survived', aspect=4)
# a.map(sns.kdeplot, 'Age', shade=True)
# a.set(xlim=(0, data1['Age'].max()))
# a.add_legend()
#
#
# h = sns.FacetGrid(data1, row='Sex', col='Pclass', hue='Survived')
# h.map(plt.hist, 'Age', alpha=0.75)
# h.add_legend()
#
#
# pp = sns.pairplot(data1, hue='Survived', palette='deep', size=1.2, diag_kind='kde', diag_kws=dict(shade=True), plot_kws=dict(s=10))
# pp.set(xticklabels=[])
#
# plt.show()


# def correlation_heatmap(df):
#     _, ax = plt.subplots(figsize=(14, 12))
#     coloemap = sns.diverging_palette(220, 10, as_cmap=True)
#
#     _ = sns.heatmap(
#         df.corr(),
#         cmap=coloemap,
#         square=True,
#         cbar_kws={'shrink':0.9},
#         ax=ax,
#         annot=True,
#         linewidths=0.1, vmax=1.0, linecolor='white',
#         annot_kws={'fontsize':12}
#     )
#
#     plt.title('Pearson Correlation of Feature', y=1.05, size=15)
#
#
# correlation_heatmap(data1)


MLA = [
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    gaussian_process.GaussianProcessClassifier(),

    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),

    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    neighbors.KNeighborsClassifier(),

    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),

    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    XGBClassifier()
]


cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, random_state=0)

MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD', 'MLA Time']
MLA_compare = pd.DataFrame(columns=MLA_columns)

MLA_predict = data1[Target]

row_index = 0
for alg in MLA:
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    cv_results = model_selection.cross_validate(alg, data1[data1_x_bin], data1[Target], cv=cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3

    alg.fit(data1[data1_x_bin], data1[Target])
    MLA_predict[MLA_name] = alg.predict(data1[data1_x_bin])

    row_index += 1


MLA_compare.sort_values(by=['MLA Test Accuracy Mean'], ascending=False, inplace=True)


sns.barplot(x='MLA Test Accuracy Mean', y='MLA Name', data=MLA_compare, color='pink')

plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Socre (%)')
plt.ylabel('Algorithm')
plt.show()



















