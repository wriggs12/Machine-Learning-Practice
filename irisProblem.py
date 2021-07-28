from scipy.spatial import distance

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

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

treeClassifier = tree.DecisionTreeClassifier()
treeClassifier.fit(X_train, y_train)

treePrediction = treeClassifier.predict(X_test)

print('Accuracy using a Decision Tree')
print(accuracy_score(y_test, treePrediction))

neighborClassifier = KNeighborsClassifier()
neighborClassifier.fit(X_train, y_train)

neighborPrediction = neighborClassifier.predict(X_test)

print('Accuracy using K Nearest Neighbor')
print(accuracy_score(y_test, neighborPrediction))

personalClassifier = personalKNN()
personalClassifier.fit(X_train, y_train)

personalPrediction = personalClassifier.predict(X_test)

print('Accuracy using my own K Nearest Neighbor')
print(accuracy_score(y_test, personalPrediction))
