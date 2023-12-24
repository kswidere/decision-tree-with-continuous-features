class TreeNode:
    def __init__(self, is_leaf, **kwargs):
        self.left: TreeNode = None
        self.right: TreeNode = None
        self.is_leaf: bool = is_leaf
        if is_leaf:
            self.target = kwargs["target"]
        else:
            self.test = kwargs["test"]


class DecisionTree:
    def __init__(self, train_dataset, train_targets, test_method,
                 default_target=None, possible_tests=None):
        self.train_dataset = train_dataset
        self.train_targets = train_targets
        self.test_method = test_method
        self.default_target = default_target or self.find_default_target()
        self.possible_tests = possible_tests or self.find_possible_tests()
        self.root: TreeNode = self.fit()

    def __str__(self):
        pass

    def fit(self):
        leaf_target = self.leaf_target()

        if leaf_target is not None:
            leaf = TreeNode(True, target=leaf_target)
            return leaf

        test = self.choose_test()
        root = TreeNode(False, test=test)
        default_target = self.find_default_target()
        left_dataset, left_targets = self.find_new_dataset(test, False)
        right_dataset, right_targets = self.find_new_dataset(test, True)
        possible_tests = self.possible_tests.remove(test)

        subtree_left = DecisionTree(left_dataset, left_targets,
                                    self.test_method, default_target,
                                    possible_tests)

        subtree_right = DecisionTree(right_dataset, right_targets,
                                     self.test_method, default_target,
                                     possible_tests)

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

    def choose_target(self):
        if not self.train_dataset:
            return self.default_target
        return self.find_default_target()

    def choose_test(self):
        return self.test_method.choose_test()

    def find_default_target(self):
        return max(set(self.train_targets), key=self.train_targets.count)

    def find_possible_tests(self):
        return self.test_method.find_possible_tests(self.train_dataset)

    def find_new_dataset(test, test_passed: bool):
        pass

    def predict(self, data):
        pass
