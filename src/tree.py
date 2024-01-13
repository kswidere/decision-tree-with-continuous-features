import pandas as pd
from test_methods import EqualFrequency
from copy import deepcopy


class TreeNode:
    def __init__(self, is_leaf, **kwargs):
        self.left: TreeNode = None
        self.right: TreeNode = None
        self.is_leaf: bool = is_leaf
        if is_leaf:
            self.target = kwargs["target"]
        else:
            self.test = kwargs["test"]

    def predict(self, dataset):
        predicted_targets = []
        for row in dataset:
            prediction = self.predict_row(row)
            predicted_targets.append(prediction)
        return predicted_targets

    def predict_row(self, row):
        if self.is_leaf:
            return self.target
        elif self.test(row):
            return self.left.predict_row(row)
        else:
            return self.right.predict_row(row)


class DecisionTree:
    def __init__(self, train_dataset, train_targets, test_method,
                 default_target=None, possible_tests=None):
        self.train_dataset = train_dataset
        self.train_targets = train_targets
        self.test_method = test_method
        self.default_target = (self.find_default_target()
                               if default_target is None else default_target)
        self.possible_tests = (self.find_possible_tests()
                               if possible_tests is None else possible_tests)
        self.root: TreeNode = self.fit()

    def __str__(self):
        pass

    def fit(self):
        leaf_target = self.leaf_target()

        if leaf_target is not None:
            leaf = TreeNode(True, target=leaf_target)
            return leaf

        test = self.choose_test()
        root = TreeNode(False, test=deepcopy(test))
        default_target = self.find_default_target()
        left_dataset, left_targets = self.find_new_dataset(test, False)
        right_dataset, right_targets = self.find_new_dataset(test, True)

        subtree_left = DecisionTree(left_dataset, left_targets,
                                    self.test_method, default_target,
                                    self.possible_tests)

        subtree_right = DecisionTree(right_dataset, right_targets,
                                     self.test_method, default_target,
                                     self.possible_tests)

        root.left = subtree_left.root
        root.right = subtree_right.root
        return root

    def leaf_target(self):
        if len(set(self.train_targets)) == 1:
            return self.train_targets[0]
        elif not self.train_dataset:
            return self.default_target
        elif not self.possible_tests:
            return self.find_default_target()
        return None

    def choose_test(self):
        return self.test_method.choose_test(self.train_dataset)

    def find_default_target(self):
        return max(set(self.train_targets), key=self.train_targets.count)

    def find_possible_tests(self):
        return self.test_method.find_possible_tests(self.train_dataset)

    def find_new_dataset(self, test, test_passed: bool):
        new_dataset = deepcopy(self.train_dataset)
        new_targets = deepcopy(self.train_targets)
        for idx, row in enumerate(self.train_dataset):
            if test(row) is not test_passed:
                del new_targets[new_dataset.index(row)]
                new_dataset.remove(row)
        return (new_dataset, new_targets)

    def predict(self, dataset):
        return self.root.predict(dataset)


if __name__ == "__main__":
    data = pd.read_csv('src/diabetes.csv')

    # Podział danych na cechy i etykiety
    features = data.drop('Outcome', axis=1)
    labels = data['Outcome']

    # Konwersja na listy
    features_list = features.values.tolist()
    labels_list = labels.values.tolist()

    tree = DecisionTree(deepcopy(features_list),
                        deepcopy(labels_list),
                        EqualFrequency())
    predictions = tree.predict(features_list)
    print(predictions)
