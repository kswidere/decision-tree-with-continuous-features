from math import log2
import math
import pandas as pd
from copy import deepcopy
import numpy as np


class TestMethod:
    def choose_test(self, train_dataset):
        """
        Wybiera najlepszy test dla danego zbioru danych.

        Args:
            dataset: Zbiór danych w postaci pandas DataFrame z kolumną 'target'.

        Returns:
            Funkcja testująca, która przyjmuje wiersz danych i zwraca wartość logiczną.
        """
        raise NotImplementedError(
            "This method needs to be overridden in subclasses")

    def find_possible_tests(self, train_dataset):
        """
        Znajduje możliwe testy dla danego zbioru danych.

        Args:
            dataset: Zbiór danych w postaci pandas DataFrame z kolumną 'target'.

        Returns:
            Lista funkcji testujących, które przyjmują wiersz danych i zwracają wartość logiczną.
        """
        raise NotImplementedError(
            "This method needs to be overridden in subclasses")


class InformationGainTest:
    def entropy(self, targets):
        value, counts = np.unique(targets, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -sum(probabilities * np.log2(probabilities))
        return entropy if not np.isnan(entropy) else 0

    def information_gain(self, left_targets, right_targets, current_entropy):
        p_left = len(left_targets) / (len(left_targets) + len(right_targets))
        p_right = 1 - p_left
        gain = current_entropy - p_left * \
            self.entropy(left_targets) - p_right * self.entropy(right_targets)
        return gain

    def find_possible_tests(self, data):
        possible_tests = []
        for feature in data.columns:
            unique_values = sorted(data[feature].unique())
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                possible_tests.append(lambda row, threshold=threshold,
                                      feature=feature: row[feature] <= threshold)
        return possible_tests

    def choose_test(self, data):
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


class EqualWidth(TestMethod):
    pass


class KMeans(TestMethod):
    pass


class GiniImpurityTest(TestMethod):
    def choose_test(self, train_dataset):
        best_test = None
        best_gini = float('inf')

        for test in self.find_possible_tests(train_dataset):
            gini = self.calculate_gini(train_dataset, test)
            if gini < best_gini:
                best_gini = gini
                best_test = test
        print(f"wybieram test {best_test}")
        return best_test

    def find_possible_tests(self, train_dataset):
        possible_tests = []

        for column in train_dataset.columns:
            if column != 'target':
                sorted_values = sorted(train_dataset[column].unique())
                for i in range(len(sorted_values) - 1):
                    average = (sorted_values[i] + sorted_values[i + 1]) / 2
                    possible_tests.append(lambda row, threshold=average,
                                          feature=column: row[feature] > threshold)
        print("found possible tests")
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
        """
        Oblicza współczynnik Gini dla danych liczb wystąpień klas.
        Zwraca target_counts: Liczby wystąpień klas w postaci pandas Series.
        """
        gini = 1 - sum((count / target_counts.sum())
                       ** 2 for count in target_counts)
        return gini


class EqualFrequencyTest(TestMethod):
    def choose_test(self, train_dataset):
        best_test = None
        best_info_gain = -np.inf

        possible_tests = self.find_possible_tests(train_dataset)

        for test in possible_tests:
            info_gain = self.calculate_information_gain(test, train_dataset)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_test = test
        print(f"wybieram test {best_test}")
        return best_test

    def find_possible_tests(self, train_dataset):
        possible_tests = []

        for column in train_dataset.columns:
            if column != 'target':
                # Sort the values and split into bins with approximately equal number of observations
                sorted_values = sorted(train_dataset[column].dropna().unique())
                n_bins = int(np.sqrt(len(sorted_values)))
                bins = pd.qcut(sorted_values, q=n_bins,
                               duplicates='drop', retbins=True)[1]

                # Create tests for the thresholds between bins
                for i in range(1, len(bins)):
                    threshold = bins[i]
                    possible_tests.append(
                        lambda row, column=column, threshold=threshold: row[column] <= threshold)
        print("found possible tests")
        return possible_tests

    def calculate_information_gain(self, test, dataset):
        entropy_before = self.calculate_entropy(dataset['target'])

        passed = dataset[dataset.apply(test, axis=1)]
        failed = dataset[~dataset.apply(test, axis=1)]

        entropy_after = (len(passed) / len(dataset)) * self.calculate_entropy(passed['target']) + \
                        (len(failed) / len(dataset)) * \
            self.calculate_entropy(failed['target'])

        return entropy_before - entropy_after

    def calculate_entropy(self, series):
        probabilities = series.value_counts(normalize=True)
        return -np.sum(probabilities * np.log2(probabilities))
