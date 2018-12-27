import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


#数据预处理
def featureHandle(pdData):
    # 删除多余不需要的列
    pdData = pdData.drop(columns=["Ticket"])
    pdData = pdData.drop(columns=["Name"])
    pdData = pdData.drop(columns=["Cabin"])
    passengerIdCol = pdData["PassengerId"]
    pdData = pdData.drop(columns=["PassengerId"])
    print(pdData.columns)

    # 字符串类别处理
    print(pdData["Sex"].unique())
    pdData["Sex"] = pdData["Sex"].fillna(pdData["Sex"].max())
    pdData.loc[pdData["Sex"] == "male", "Sex"] = 0
    pdData.loc[pdData["Sex"] == "female", "Sex"] = 1

    print(pdData["Embarked"].unique())
    pdData["Embarked"] = pdData["Embarked"].fillna("S")
    pdData.loc[pdData["Embarked"] == "S", "Embarked"] = 0
    pdData.loc[pdData["Embarked"] == "C", "Embarked"] = 1
    pdData.loc[pdData["Embarked"] == "Q", "Embarked"] = 2

    #缺失值处理
    pdData["Age"] = pdData["Age"].fillna(pdData["Age"].median())

    pdData["Fare"] = pdData["Fare"].fillna(pdData["Fare"].median())

    print(pdData.isnull().any())

    return pdData,passengerIdCol


def loadData():
    pd_train = pd.read_csv("train.csv")
    pd_test = pd.read_csv("test.csv")

    pd_train,train_pass_id = featureHandle(pd_train)
    pd_test,test_pass_id = featureHandle(pd_test)

    nd_train = pd_train.values
    nd_test = pd_test.values

    train_X = nd_train[:,1:]
    train_Y = nd_train[:,0]

    return train_X,train_Y,nd_test,test_pass_id

def resultsAndSaveHandle(results,test_pass_id):
    results = results.astype(np.int)
    results = pd.Series(results, name="Survived")

    submission = pd.concat([test_pass_id, results], axis=1)

    print(submission["Survived"].value_counts())

    submission.to_csv("titanic_simple_lr_submission.csv", index=False)

def titanic_simple_handle():
    train_X,train_Y,test_X,test_pass_id= loadData()
    print(test_X.shape)

    lg_clf = LogisticRegression()
    # results = cross_val_score(lg_clf,train_X,train_Y,cv=5)
    # print(results)
    # print(results.mean())

    lg_clf.fit(train_X,train_Y)
    results = lg_clf.predict(test_X)

    resultsAndSaveHandle(results,test_pass_id)




if __name__ == "__main__":
    titanic_simple_handle()

