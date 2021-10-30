import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

def euc(a, b):
    return distance.euclidean(a, b)

class personalKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []

        for row in X_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, row):
        smallestDist = euc(row, self.X_train[0])
        index = 0

        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < smallestDist:
                smallestDist = dist
                index = i

        return self.y_train[index]

class personalKMean():
    def __init__(self, k = 3, tol = 0.0001, maxIterations = 300):
        self.k = k
        self.tol = tol
        self.maxIterations = maxIterations

    def fit(self, data):
        self.clusters = {}

        for i in range(self.k):
            self.clusters[i] = data[i]

        for i in range(self.maxIterations):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for point in data:
                distances = [np.linalg.norm(point - self.clusters[cluster]) for cluster in self.clusters]
                classification = distances.index(min(distances))
                self.classifications[classification].append(point)

            prevClusters = dict(self.clusters)

            for group in self.classifications:
                self.clusters[group] = np.average(self.classifications[group], axis=0)

            optimized = True

            for cluster in self.clusters:
                originalCluster = prevClusters[cluster]
                curCluster = self.clusters[cluster]
                if np.sum((curCluster - originalCluster) / originalCluster * 100.0) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.clusters[cluster]) for cluster in self.clusters]
        classification = distances.index(min(distances))
        return classification

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

#1 Decision Tree Classifier
treeClassifier = tree.DecisionTreeClassifier()
treeClassifier.fit(X_train, y_train)

treePrediction = treeClassifier.predict(X_test)

print('Accuracy using a Decision Tree')
print(accuracy_score(y_test, treePrediction))

#2 K-Nearest Neighbors Classifier
personalClassifier = personalKNN()
personalClassifier.fit(X_train, y_train)

personalPrediction = personalClassifier.predict(X_test)

print('Accuracy using my own K Nearest Neighbor Model')
print(accuracy_score(y_test, personalPrediction))

#3 K-Means Classifier
personalMeanClassifier = personalKMean()
personalMeanClassifier.fit(X_train)

correct = 0
for i in range(len(X_test)):
    dataPoint = X_test[i]
    prediction = personalMeanClassifier.predict(dataPoint)

    if prediction == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)

print('Accuracy using my own K Means Clustering Model')
print(accuracy)