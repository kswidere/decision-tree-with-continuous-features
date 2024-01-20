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
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score
from time import time
import matplotlib.pyplot as plt


def prepare_data(csv_path: str, target_name: str):
    data = pd.read_csv(csv_path)
    data = data.rename(columns={target_name: 'target'})
    return data


def predict_using_sklearn(train_data, test_data, criterion='entropy'):
    tree = sktree.DecisionTreeClassifier(criterion=criterion)
    tree.fit(train_data.drop(columns='target'), train_data['target'])
    y_pred = tree.predict(test_data.drop(columns='target'))
    return y_pred


# def run_experiment(data, test_method, num_runs=2):
#     results = []
#     for _ in range(num_runs):
#         train_data, test_data = train_test_split(data, test_size=0.2)
#         tree = DecisionTree(deepcopy(train_data),
#                             test_method=test_method, max_height=12)
#         y_pred = tree.predict(test_data)
#         results.append(classification_report(test_data['target'], y_pred))

#     return results


# def summarize_results(results):
#     scores = [float(result.split('accuracy')[1].split()[0])
#               for result in results]
#     return np.mean(scores), np.std(scores), max(scores), min(scores)


def run_experiments(data, test_method, num_runs=2, num_params=2):
    results = {'accuracy': [],
               'recall': [], 'precision': [], 'time': []}
    for param in range(2, num_params+2):
        accuracies, recalls, precisions, times = [], [], [], []
        for _ in range(num_runs):
            train_data, test_data = train_test_split(data, test_size=0.2)
            start_time = time()
            tree = DecisionTree(deepcopy(train_data),
                                test_method=test_method(param), max_height=12)
            elapsed_time = time() - start_time
            y_pred = tree.predict(test_data)
            accuracies.append(accuracy_score(test_data['target'], y_pred))
            recalls.append(recall_score(
                test_data['target'], y_pred, average='macro'))
            precisions.append(precision_score(
                test_data['target'], y_pred, average='macro'))
            times.append(elapsed_time)
        results['accuracy'].append(np.mean(accuracies))
        results['recall'].append(np.mean(recalls))
        results['precision'].append(np.mean(precisions))
        results['time'].append(np.mean(times))
    return results


def plot_results(results, title):
    params = range(2, len(results['accuracy']) + 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.scatter(params, results['accuracy'], label='Accuracy')
    ax1.scatter(params, results['recall'], label='Recall')
    ax1.scatter(params, results['precision'], label='Precision')
    ax1.set_title(f"{title} method")
    ax1.set_xlabel('Parameter')
    ax1.set_ylabel('Score')
    ax1.legend()

    ax2.scatter(params, results['time'], label='Time', color='r')
    ax2.set_title('Accuracy, Recall, Precision, Time vs Parameter')
    ax2.set_xlabel('Parameter')
    ax2.set_ylabel('Time (s)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"plots/{title}.png")
    plt.show()


if __name__ == "__main__":
    iris_data = prepare_data('src/Iris.csv', 'Species')
    diabetes_data = prepare_data('src/diabetes.csv', 'Outcome')
    test_methods = [EqualWidth, EqualFrequency, KMeansTest]
    plot_results(run_experiments(diabetes_data, EqualWidth), 'Equal Width')
    plot_results(run_experiments(
        diabetes_data, EqualFrequency), 'Equal Frequency')
    plot_results(run_experiments(diabetes_data, KMeansTest), 'KMeans')
    # test_methods = [EqualWidth(num_intervals=2), EqualFrequency(
    #     num_groups=2), KMeansTest(num_clusters=2)]
    # results = {}

    # for test_method in test_methods:
    #     results[test_method.__class__.__name__] = run_experiment(
    #         diabetes_data, test_method)

    # for test_method, result in results.items():
    #     mean, std, best, worst = summarize_results(result)
    #     print(f"Wyniki dla {test_method}:")
    #     print(f"Średnia: {mean}")
    #     print(f"Odchylenie standardowe: {std}")
    #     print(f"Najlepszy wynik: {best}")
    #     print(f"Najgorszy wynik: {worst}")
    #     print()

    # train_data, test_data = train_test_split(diabetes_data, test_size=0.2)
    # tree = DecisionTree(deepcopy(train_data), test_method=InformationGain())
    # y_pred = tree.predict(test_data)
    # print(classification_report(test_data['target'], y_pred))
    # y_pred_sklearn = predict_using_sklearn(train_data, test_data)
    # print(classification_report(test_data['target'], y_pred_sklearn))
