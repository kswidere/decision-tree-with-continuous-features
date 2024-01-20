"""
Autorzy: Jakub Kowalczyk, Kinga Świderek
"""
import pandas as pd
from test_methods import (InformationGain, EqualFrequency, GiniImpurity,
                          EqualWidth, KMeansTest)
from copy import deepcopy
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import tree as sktree
from tree import DecisionTree


def prepare_data(csv_path: str, target_name: str):
    data = pd.read_csv(csv_path)
    data = data.rename(columns={target_name: 'target'})
    return data


def predict_using_sklearn(train_data, test_data, criterion='entropy'):
    tree = sktree.DecisionTreeClassifier(criterion=criterion)
    tree.fit(train_data.drop(columns='target'), train_data['target'])
    y_pred = tree.predict(test_data.drop(columns='target'))
    return y_pred


if __name__ == "__main__":
    iris_data = prepare_data('src/Iris.csv', 'Species')
    diabetes_data = prepare_data('src/diabetes.csv', 'Outcome')
    train_data, test_data = train_test_split(diabetes_data, test_size=0.2)
    tree = DecisionTree(deepcopy(train_data), test_method=InformationGain())
    y_pred = tree.predict(test_data)
    print(classification_report(test_data['target'], y_pred))
    # y_pred_sklearn = predict_using_sklearn(train_data, test_data)
    # print(classification_report(test_data['target'], y_pred_sklearn))
