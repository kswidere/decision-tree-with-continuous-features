from experiments import prepare_data, evaluate_params, plot_params_evaluation
from test_methods import EqualFrequency, EqualWidth, KMeansTest

iris_data = prepare_data('Iris.csv', 'Species')
diabetes_data = prepare_data('diabetes.csv', 'Outcome')

iris_eq_freq = evaluate_params(iris_data, EqualFrequency)
iris_eq_width = evaluate_params(iris_data, EqualWidth)
iris_k_means = evaluate_params(iris_data, KMeansTest)

plot_params_evaluation(iris_eq_freq, "Equal Frequency for iris dataset")
plot_params_evaluation(iris_eq_width, "Equal Width for iris dataset")
plot_params_evaluation(iris_k_means, "K-means for iris dataset")

diabetes_eq_freq = evaluate_params(diabetes_data, EqualFrequency)
diabetes_eq_width = evaluate_params(diabetes_data, EqualWidth)
diabetes_k_means = evaluate_params(diabetes_data, KMeansTest)

plot_params_evaluation(diabetes_eq_freq, "Equal Frequency for diabetes dataset")
plot_params_evaluation(diabetes_eq_width, "Equal Width for diabetes dataset")
plot_params_evaluation(diabetes_k_means, "K-means for diabetes dataset")
