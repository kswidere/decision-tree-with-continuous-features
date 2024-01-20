"""
Autor: Jakub Kowalczyk
"""
from math import log2
import pandas as pd
from copy import deepcopy
import numpy as np
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod


class TestMethod(ABC):
    @abstractmethod
    def find_possible_tests(self, data: pd.DataFrame):
        """
        Znajduje możliwe testy dla danego zbioru danych.

        Args:
            dataset: Zbiór danych w postaci pandas DataFrame z kolumną 'target'.

        Returns:
            Lista funkcji testujących, które przyjmują wiersz danych i zwracają wartość logiczną.
        """
        pass

    def choose_test(self, data: pd.DataFrame):
        """
        Wybiera najlepszy test dla danego zbioru danych, bazując na information gain.

        Args:
            dataset: Zbiór danych w postaci pandas DataFrame z kolumną 'target'.

        Returns:
            Funkcja testująca, która przyjmuje wiersz danych i zwraca wartość logiczną.
        """
        targets = data['target']
        current_entropy = self.entropy(targets)
        best_gain = -1
        best_test = None

        for test in self.find_possible_tests(data.drop(columns='target')):
            left_targets = targets[data.drop(
                columns='target').apply(test, axis=1)]
            right_targets = targets[~data.drop(
                columns='target').apply(test, axis=1)]
            gain = self.information_gain(
                left_targets, right_targets, current_entropy)

            if gain > best_gain:
                best_gain = gain
                best_test = test

        return best_test

    def entropy(self, targets):
        _, counts = np.unique(targets, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -sum(probabilities * np.log2(probabilities))
        return entropy if not np.isnan(entropy) else 0

    def information_gain(self, left_targets, right_targets, current_entropy):
        p_left = len(left_targets) / (len(left_targets) + len(right_targets))
        p_right = 1 - p_left
        gain = current_entropy - p_left * \
            self.entropy(left_targets) - p_right * self.entropy(right_targets)
        return gain


class InformationGain(TestMethod):
    def __str__(self) -> str:
        return 'Information Gain'

    def find_possible_tests(self, data):
        possible_tests = []
        for feature in data.columns[:-1]:
            unique_values = sorted(data[feature].unique())
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                possible_tests.append(lambda row, threshold=threshold,
                                      feature=feature: row[feature] <= threshold)
        return possible_tests


class EqualFrequency(TestMethod):
    def __init__(self, n=3):
        self.num_groups = n

    def __str__(self) -> str:
        return f'EqualFrequency(num_groups={self.num_groups})'

    def find_possible_tests(self, data: pd.DataFrame):
        tests = []

        for column in data.columns:
            sorted_values = data[column].sort_values()
            num_rows = len(sorted_values)
            group_size = num_rows // self.num_groups

            for i in range(self.num_groups):
                lower_index = i * group_size
                upper_index = (
                    i + 1) * group_size if i < self.num_groups - 1 else num_rows
                lower_bound = sorted_values.iloc[lower_index]
                upper_bound = sorted_values.iloc[upper_index - 1]

                def test(row, column=column, lower_bound=lower_bound, upper_bound=upper_bound):
                    return lower_bound <= row[column] <= upper_bound

                tests.append(test)

        return tests


class EqualWidth(TestMethod):
    def __init__(self, n=3):
        self.num_intervals = n

    def __str__(self) -> str:
        return f'EqualWidth(num_intervals={self.num_intervals})'

    def find_possible_tests(self, data: pd.DataFrame):
        tests = []

        for column in data.columns[:-1]:
            min_value = data[column].min()
            max_value = data[column].max()
            interval_width = (max_value - min_value) / self.num_intervals

            for i in range(self.num_intervals):
                lower_bound = min_value + i * interval_width
                upper_bound = lower_bound + interval_width

                def test(row, column=column, lower_bound=lower_bound, upper_bound=upper_bound):
                    return lower_bound <= row[column] < upper_bound

                tests.append(test)
        return tests


class KMeansTest(TestMethod):
    def __init__(self, n=3):
        self.num_clusters = n

    def __str__(self) -> str:
        return f'KMeansTest(num_clusters={self.num_clusters})'

    def find_possible_tests(self, data: pd.DataFrame):
        tests = []
        # Exclude the last column which contains the class
        for column in data.columns[:-1]:
            values = data[column].values.reshape(-1, 1)
            num_clusters = min(self.num_clusters, len(np.unique(values)))
            if num_clusters > 1:  # Ensure there are at least 2 clusters to avoid ValueError
                kmeans = KMeans(n_clusters=num_clusters,
                                random_state=0).fit(values)
                cluster_centers = sorted(kmeans.cluster_centers_.flatten())

                for i in range(1, len(cluster_centers)):
                    threshold = (
                        cluster_centers[i - 1] + cluster_centers[i]) / 2
                    tests.append(lambda x, column=column,
                                 threshold=threshold: x[column] < threshold)
        return tests


class GiniImpurity(TestMethod):
    def __str__(self) -> str:
        return 'Gini Impurity'

    def choose_test(self, train_dataset):
        self.possible_tests = self.find_possible_tests(train_dataset)

        best_test = None
        best_gini = float('inf')

        for test in self.possible_tests:
            gini = self.calculate_gini(train_dataset, test)
            if gini < best_gini:
                best_gini = gini
                best_test = test
        return best_test

    def find_possible_tests(self, train_dataset):
        possible_tests = []

        for column in train_dataset.columns[:-1]:
            sorted_values = sorted(train_dataset[column].unique())
            for i in range(len(sorted_values) - 1):
                average = (sorted_values[i] + sorted_values[i + 1]) / 2
                possible_tests.append(lambda row, threshold=average,
                                      feature=column: row[feature] > threshold)
        return possible_tests

    def calculate_gini(self, train_dataset, test):
        left_data = train_dataset[train_dataset.apply(test, axis=1)]
        right_data = train_dataset[~train_dataset.apply(test, axis=1)]

        left_target_counts = left_data['target'].value_counts()
        right_target_counts = right_data['target'].value_counts()

        left_gini = self.calculate_gini_for_target_counts(left_target_counts)
        right_gini = self.calculate_gini_for_target_counts(right_target_counts)

        return (len(left_data) / len(train_dataset)) * left_gini + (len(right_data) / len(train_dataset)) * right_gini

    def calculate_gini_for_target_counts(self, target_counts):
        gini = 1 - sum((count / target_counts.sum())
                       ** 2 for count in target_counts)
        return gini  # współczynnik Gini dla danych liczb wystąpień klas.
