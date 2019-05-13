
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, eta, epochs):
        self.eta = eta
        self.epochs = epochs

    def __threshold_array__(self, net_sums):
        threshold_function = lambda x: 1 if x >= 0 else -1
        return np.array([threshold_function(x) for x in net_sums])

    def __threshold_number__(self, num):
        threshold_function = lambda x: 1 if x >= 0 else -1
        return threshold_function(num)

    def fit(self, x_train, y_labels):
        """
        trains the perceptron given the training set and the actual labels for every instance of the set.
        The result of the training is the weight coefficients which are used in the predict() function.
        :param x_train: np array of the training set with rows as samples
        :param y_labels: np array for every sample in the training set
        :return: None
        """
        if x_train.shape[0] != y_labels.shape[0]:
            raise ValueError(
                "x_train [{}] and y_labels [{}] array dimensions don't agree".format(
                    x_train.shape, y_labels.shape
                )
            )
        # adds a column of ones to the left of the x_train matrix and then transposes it such that
        # the samples become columns instead of rows
        x_train_with_bias = np.hstack((np.ones((x_train.shape[0], 1), dtype=x_train.dtype), x_train)).transpose()

        # weights is a with 1 + Ndims elements
        self.weights = np.random.normal(loc = 0, scale = 1, size = x_train_with_bias.shape[0])
        self.weights_history = np.array([self.weights])
        for epoch in range(self.epochs):
            y_current = self.__threshold_array__(np.dot(self.weights, x_train_with_bias))
            self.weights = self.weights + np.dot(self.eta * (y_labels - y_current), x_train_with_bias.transpose())
            self.weights_history = np.append(self.weights_history, np.array([self.weights]), axis=0)

        print("Training completed successfully, weights: [{}]".format(self.weights))

    def predict(self, x):
        """
        Predicts if a sample belongs to a "-1" or "+1" class
        :param x: np array as a row for a single sample
        :return: -1 or +1
        """
        if x.shape[0] + 1 != self.weights.shape[0]:
            raise ValueError(
                "Input has wrong shape [{}], expected [{}]".format(
                    x.shape[0],
                    self.weights.shape[0] - 1
                )
            )
        x_with_bias = np.concatenate((np.ones(shape = (1,)), x))
        return self.__threshold_number__(np.dot(self.weights, x_with_bias))

class Evaluation:
    def __init__(self, eta, epochs):
        self.perceptron = Perceptron(eta, epochs)

    def lineraly_separable(self):
        minus_one = np.random.normal(loc=0, scale=1, size=(10,2))
        plus_one = np.random.normal(loc=3, scale=1, size=(10,2))
        x_train = np.concatenate((minus_one, plus_one))

        y_lables = np.array([-1 for _ in range(10)] + [1 for _ in range(10)])

        self.perceptron.fit(x_train=x_train, y_labels=y_lables)

        plt.plot(minus_one[:, 0], minus_one[:, 1], 'ro')
        plt.plot(plus_one[:, 0], plus_one[:, 1], 'go')
        x = np.linspace(-5, 10, 100)
        y = -(self.perceptron.weights[0] + x*self.perceptron.weights[2])/self.perceptron.weights[1]
        plt.plot(x, y, label='decision boundary')
        plt.legend(loc='upper left')
        plt.show()
        return x_train, y_lables

    def viz_weights_history(self, x_train, y_label):
        plt.plot(x_train[0:10, 0], x_train[0:10, 1], 'ro')
        plt.plot(x_train[10:20, 0], x_train[10:20, 1], 'go')
        #plt.plot(plus_one[:, 0], plus_one[:, 1], 'go')
        for curr_weight in range(self.perceptron.weights_history.shape[0]):
            x = np.linspace(-5, 10, 100)
            y = -(self.perceptron.weights_history[curr_weight, 0] + x*self.perceptron.weights_history[curr_weight, 2])/self.perceptron.weights_history[curr_weight, 1]
            plt.plot(x, y, label="bound(epoch={})".format(curr_weight))
        plt.legend(loc='upper left')
        plt.show()


eval = Evaluation(eta=0.005, epochs=10)
x_train, y_label = eval.lineraly_separable()
eval.viz_weights_history(x_train, y_label)
