import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class naiveBayesClassifier():
    def fit(self, features, labels):
        numSamples, numFeatures = features.shape
        self.uniqueLabels = np.unique(labels)
        numLabels = len(self.uniqueLabels)

        self.mean = np.zeros((numLabels, numFeatures), dtype=np.float64)
        self.variance = np.zeros((numLabels, numFeatures), dtype=np.float64)
        self.prevProb = np.zeros(numLabels, dtype=np.float64)

        for label in self.uniqueLabels:
            curLabel = features[label==labels]
            self.mean[label, :] = curLabel.mean(axis=0)
            self.variance[label, :] = curLabel.var(axis=0)
            self.prevProb[label] = curLabel.shape[0] / float(numSamples)

    def predict(self, data):
        predictions = [self.predictHelper(row) for row in data]
        return np.array(predictions).astype(int)

    def predictHelper(self, x):
        probabilities = []

        for index, label in enumerate(self.uniqueLabels):
            prior = np.log(self.prevProb[index])
            probability = np.sum(np.log(self.pdf(index, x)))
            probability = prior + probability
            probabilities.append(probability)

        return self.uniqueLabels[np.argmax(probabilities)]

    def pdf(self, index, x):
        curMean = self.mean[index]
        curVariance = self.variance[index]
        numerator = np.exp(-1 * ((x - curMean) ** 2) / (2 * curVariance))
        demoninator = np.sqrt(2 * np.pi * curVariance)

        return numerator / demoninator

trainData = pd.read_csv('./Data/titanic_train.csv')
predictData = pd.read_csv('./Data/titanic_test.csv')

trainData.drop(['PassengerId', 'Name', 'SibSp', ], axis='columns', inplace=True)
predictData.drop(['PassengerId', 'Name', 'SibSp', ], axis='columns', inplace=True)

labeler = LabelEncoder()
for column in trainData[
    ["Sex", "Ticket", "Cabin", "Embarked"]].columns:
    trainData[column] = labeler.fit_transform(trainData[column].values)
for column in predictData[
    ["Sex", "Ticket", "Cabin", "Embarked"]].columns:
    predictData[column] = labeler.fit_transform(predictData[column].values)

target = trainData.Survived
trainData.drop('Survived', axis='columns', inplace=True)

input = trainData

trainData.Age = trainData.Age.fillna(trainData.Age.mean())
predictData.Age = predictData.Age.fillna(predictData.Age.mean())

X_train, X_test, y_train, y_test = train_test_split(input, target, test_size=0.2)

classifier = naiveBayesClassifier()
classifier.fit(X_train.to_numpy(), y_train.to_numpy())

print(accuracy_score(y_test.to_numpy(), classifier.predict(X_test.to_numpy())))