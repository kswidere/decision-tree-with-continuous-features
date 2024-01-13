class TestMethod:
    def choose_test(self):
        raise NotImplementedError(
            "This method needs to be overridden in subclasses")

    def find_possible_tests(self, train_dataset):
        raise NotImplementedError(
            "This method needs to be overridden in subclasses")


class EqualFrequency(TestMethod):
    def __init__(self):
        self.possible_tests = []

    def choose_test(self, train_dataset):
        if not self.possible_tests:
            self.find_possible_tests(train_dataset)
        return self.possible_tests.pop()

    def find_possible_tests(self, train_dataset):
        # Sortujemy dane
        # zakładamy, że testujemy pierwszy atrybut
        sorted_dataset = sorted(train_dataset, key=lambda x: x[0])
        # Dzielimy dane na dwie równoliczne grupy
        mid_index = len(sorted_dataset) // 2
        # Tworzymy test na podstawie mediany
        threshold = (sorted_dataset[mid_index - 1]
                     [0] + sorted_dataset[mid_index][0]) / 2
        self.possible_tests.append(lambda x: x[0] <= threshold)

# class EqualFrequency(TestMethod):
#     def choose_test(self, train_dataset):
#         # Assuming train_dataset is a list of numerical values
#         sorted_dataset = sorted(train_dataset)
#         median_index = len(sorted_dataset) // 2
#         threshold = sorted_dataset[median_index]

#         def test(row):
#             return row <= threshold

#         return test

#     def find_possible_tests(self, train_dataset):
#         # Assuming train_dataset is a list of lists of numerical values
#         possible_tests = []
#         for feature_index in range(len(train_dataset[0])):
#             # Extract the feature column
#             feature_values = [row[feature_index] for row in train_dataset]
#             # Sort the feature values
#             sorted_feature_values = sorted(feature_values)
#             # Find the median index
#             median_index = len(sorted_feature_values) // 2
#             # Use the median value as the threshold
#             threshold = sorted_feature_values[median_index]
#             # Define the test function

#             def test(row, threshold=threshold, feature_index=feature_index):
#                 return row[feature_index] <= threshold
#             # Add the test function to the list of possible tests
#             possible_tests.append(test)
#         return possible_tests


class EqualWidth(TestMethod):
    pass


class KMeans(TestMethod):
    pass


class Gini(TestMethod):
    pass


class InfGain(TestMethod):
    pass
