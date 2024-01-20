"""
Autorzy: Jakub Kowalczyk, Kinga Świderek
"""
import pandas as pd
from test_methods import (InformationGain, EqualFrequency, GiniImpurity,
                          EqualWidth, KMeansTest)
from copy import deepcopy
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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


def evaluate_params(data, test_method, num_runs=2, num_params=35):
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


def plot_params_evaluation(results, title):
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


def evaluate_results(data, test_method, num_runs=2, test_method_params=None):
    num_classes = data['target'].nunique()
    results = []

    method_results = {
        'error': [],
        'accuracy': [],
        'recall_macro': [],
        'recall_micro': [],
        'precision_macro': [],
        'precision_micro': [],
        'f_measure_macro': [],
        'f_measure_micro': [],
        'confusion_matrix': np.zeros((num_classes, num_classes))
    }

    for _ in range(num_runs):
        train_data, test_data = train_test_split(data, test_size=0.2)

        if test_method_params is not None:  # discretization methods
            test_method_instance = test_method(**test_method_params)
            model = DecisionTree(
                deepcopy(train_data), test_method=test_method_instance, max_height=12)
        else:                              # gini impurity and inf gain
            test_method_instance = test_method()
            model = DecisionTree(
                deepcopy(train_data), test_method=test_method_instance)

        predictions = model.predict(test_data)
        method_results['error'].append(
            1 - accuracy_score(test_data['target'], predictions))
        method_results['accuracy'].append(
            accuracy_score(test_data['target'], predictions))
        method_results['confusion_matrix'] += confusion_matrix(
            test_data['target'], predictions)

        if num_classes > 2:  # multi-class classification
            method_results['recall_macro'].append(
                recall_score(test_data['target'], predictions, average='macro'))
            method_results['recall_micro'].append(
                recall_score(test_data['target'], predictions, average='micro'))
            method_results['precision_macro'].append(
                precision_score(test_data['target'], predictions, average='macro'))
            method_results['precision_micro'].append(
                precision_score(test_data['target'], predictions, average='micro'))
            method_results['f_measure_macro'].append(
                f1_score(test_data['target'], predictions, average='macro'))
            method_results['f_measure_micro'].append(
                f1_score(test_data['target'], predictions, average='micro'))
        else:  # binary classification
            method_results['recall_macro'].append(
                recall_score(test_data['target'], predictions))
            method_results['precision_macro'].append(
                precision_score(test_data['target'], predictions))
            method_results['f_measure_macro'].append(
                f1_score(test_data['target'], predictions))

    for metric in method_results:
        if len(method_results[metric]) > 0 and metric != 'confusion_matrix':
            results.append({
                'metric': metric,
                'mean': round(np.mean(method_results[metric]), 2),
                'std': round(np.std(method_results[metric]), 2),
                'max': round(np.max(method_results[metric]), 2),
                'min': round(np.min(method_results[metric]), 2)
            })
        elif metric == 'confusion_matrix':
            method_results[metric] /= num_runs  # Average the confusion matrix

    print(f"\nResults for {test_method_instance}:")
    results_df = pd.DataFrame(results)

    print(results_df.to_latex(index=False))
    print(f"Confusion matrix:\n{method_results['confusion_matrix']}")


if __name__ == "__main__":
    iris_data = prepare_data('src/Iris.csv', 'Species')
    diabetes_data = prepare_data('src/diabetes.csv', 'Outcome')
    evaluate_results(diabetes_data, EqualFrequency, 2, {'n': 3})
    evaluate_results(diabetes_data, EqualWidth, 2, {'n': 3})
    evaluate_results(diabetes_data, KMeansTest, 2, {'n': 3})
    evaluate_results(diabetes_data, GiniImpurity, 2)
    evaluate_results(diabetes_data, InformationGain, 2)

    # plot_results(run_experiments(diabetes_data, EqualWidth), 'Equal Width')
    # plot_results(run_experiments(
    #     diabetes_data, EqualFrequency), 'Equal Frequency')
    # plot_results(run_experiments(diabetes_data, KMeansTest), 'KMeans')

    # train_data, test_data = train_test_split(diabetes_data, test_size=0.2)
    # y_pred_sklearn = predict_using_sklearn(train_data, test_data)
    # print(classification_report(test_data['target'], y_pred_sklearn))
