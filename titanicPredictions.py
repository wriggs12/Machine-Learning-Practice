import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class naiveBayesClassifier():
    def __init__(self):
        pass

    def fit(self, data):
        pass

    def predict(self, data):
        pass

trainData = pd.read_csv('./Data/titanic_train.csv')
testData = pd.read_csv('./Data/titanic_test.csv')

trainData.drop(['PassengerId', 'Name', 'SibSp', ], axis='columns', inplace=True)
testData.drop(['PassengerId', 'Name', 'SibSp', ], axis='columns', inplace=True)

labeler = LabelEncoder()
for column in trainData[
    ["Sex", "Ticket", "Cabin", "Embarked"]].columns:
    trainData[column] = labeler.fit_transform(trainData[column].values)
for column in testData[
    ["Sex", "Ticket", "Cabin", "Embarked"]].columns:
    trainData[column] = labeler.fit_transform(trainData[column].values)

target = trainData.Survived
trainData.drop('Survived', axis='columns', inplace=True)

trainData.Age = trainData.Age.fillna(trainData.Age.mean())
