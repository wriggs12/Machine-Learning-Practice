#import helper libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

#Split data into training and testing sets
def splitTrainTest(features, labels):
    return train_test_split(features, labels, test_size=0.25)

#Train the model using the training data
def trainModel(features, labels):
    #Split data into test and training sets
    xTrain, xTest, yTrain, yTest = splitTrainTest(features, labels)

    #Create and train decision tree classifier
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(xTrain, yTrain)

    #Predict using the test data
    predictions = classifier.predict(xTest)

    #See the accuracy of the model
    print(accuracy_score(yTest, predictions))

#Generates and displays heat map
def generateHeatMap(data):
    sns.set_style('whitegrid')
    corr = data.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr, annot=True, cmap="Reds")
    plt.title('Correlation Heatmap', fontsize=20)
    plt.show()

#Main Program
def main():
    print("Student Scores Predictions")
    #input data from csv file
    data = pd.read_csv('./Data/student-mat.csv')

    #Label all features with numbers
    labeler = LabelEncoder()
    for column in data[
        ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup",
         "paid", "activities", "nursery", "higher", "internet", "romantic"]].columns:
        data[column] = labeler.fit_transform(data[column].values)

    #Change grades to pass fail
    data.loc[data['G3'] < 10, ['G3']] = 0
    data.loc[data['G3'] >= 10, ['G3']] = 1

    data.loc[data['G2'] < 10, ['G2']] = 0
    data.loc[data['G2'] >= 10, ['G2']] = 1

    data.loc[data['G1'] < 10, ['G1']] = 0
    data.loc[data['G1'] >= 10, ['G1']] = 1

    #Seperate the features from the labels
    label = data.pop('G3')
    features = data

    print("\nModel Accuracy Knowing G1 & G2 Scores")
    print("----------------------------------------")
    trainModel(features, label)

    features.drop(['G2'], axis=1, inplace=True)
    print("\nModel Accuracy Knowing Only G1 Score")
    print("----------------------------------------")
    trainModel(features, label)

    features.drop(['G1'], axis=1, inplace=True)
    print("\nModel Accuracy Without Knowing Scores")
    print("----------------------------------------")
    trainModel(features, label)

if __name__ == '__main__':
    main()