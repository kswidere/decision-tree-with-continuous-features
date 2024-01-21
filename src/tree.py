"""
Autor: Kinga Świderek
"""

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
        for _, row in dataset.iterrows():
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
    def __init__(self, train_data, test_method,
                 default_target=None, possible_tests=None,
                 max_height=None, height=None):
        self.train_data = train_data
        self.test_method = test_method
        self.max_height = max_height
        self.height = 0 if height is None else height
        self.default_target = (self.find_default_target()
                               if default_target is None else default_target)
        self.possible_tests = (self.find_possible_tests()
                               if possible_tests is None else possible_tests)
        self.root: TreeNode = self.fit()

    def fit(self):
        self.height += 1
        leaf_target = self.leaf_target()

        if leaf_target is not None:
            leaf = TreeNode(True, target=leaf_target)
            return leaf

        test = self.choose_test()
        root = TreeNode(False, test=deepcopy(test))
        default_target = self.find_default_target()
        left_data = self.find_new_data(test, True)
        right_data = self.find_new_data(test, False)

        subtree_left = DecisionTree(left_data,
                                    self.test_method,
                                    default_target,
                                    possible_tests=self.possible_tests,
                                    max_height=self.max_height,
                                    height=self.height)
        subtree_right = DecisionTree(right_data,
                                     self.test_method,
                                     default_target,
                                     possible_tests=self.possible_tests,
                                     max_height=self.max_height,
                                     height=self.height)

        root.left = subtree_left.root
        root.right = subtree_right.root
        return root

    def leaf_target(self):
        if len(self.train_data['target'].unique()) == 1:
            return self.train_data['target'].iloc[0]
        elif self.train_data.empty:
            return self.default_target
        elif not self.possible_tests or (self.max_height and self.height >= self.max_height):
            return self.find_default_target()
        return None

    def choose_test(self):
        return self.test_method.choose_test(self.train_data)

    def find_default_target(self):
        return self.train_data['target'].mode()[0]

    def find_possible_tests(self):
        return self.test_method.find_possible_tests(self.train_data)

    def find_new_data(self, test, test_passed: bool):
        new_data = deepcopy(self.train_data)
        new_data = new_data[new_data.apply(test, axis=1) == test_passed]
        return new_data

    def predict(self, dataset):
        return self.root.predict(dataset)
