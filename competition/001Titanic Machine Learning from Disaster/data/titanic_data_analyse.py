import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
#train_df.info()

#print(train_df.head())

#Step1 初探数据
#print(train_df.describe())

#性别获救比例
sex_count = train_df.Sex[train_df.Survived == 1].value_counts() / train_df.Sex.value_counts()
#print(sex_count)
#print(df.Sex[df.Survived == 1])
#print(df.Sex[df.Survived == 1].value_counts())
#print(df.Sex.value_counts())

#仓等级获救比例
pclass_count = train_df.Pclass[train_df.Survived == 1].value_counts() / train_df.Pclass.value_counts()
#print(pclass_count)

#缺失值查询
#print(train_df.isnull().any())
#print(test_df.isnull().any())


#训练数据分析
#01是否存活
print(train_df.Survived.describe())
#train_df.Survived.value_counts().plot.pie()
train_df['Survived'].value_counts().plot.pie(autopct = '%1.2f%%')
plt.show()

print("-" * 60)

#02性别
train_df.Sex[train_df.Sex=='male'] = 0
train_df.Sex[train_df.Sex=='female'] = 1
print(train_df.Sex.describe())

