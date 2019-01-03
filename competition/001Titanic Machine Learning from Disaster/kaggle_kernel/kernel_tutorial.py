import random as rnd

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

def printLine():
    print('_' * 200)

train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')
combine = [train_df, test_df]

print(train_df.columns)
print(train_df.columns.values)
printLine()

print(train_df.head())
printLine()

print(str(train_df.info()))
printLine()
print(test_df.info())
printLine()

#选取所有类型为object的特征
print(train_df.describe(include=['O']))
printLine()

print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False)
      .mean().sort_values(by='Survived', ascending=False))
print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False)
      .mean().sort_values(by='Survived', ascending=False))
print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False)
      .mean().sort_values(by='Survived', ascending=False))
print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False)
      .mean().sort_values(by='Survived', ascending=False))
printLine()

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

plt.show()
