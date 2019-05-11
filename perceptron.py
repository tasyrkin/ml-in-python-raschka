
import numpy as np


class Perceptron:

    def __init__(self, eta, epochs):
        self.eta = eta
        self.epochs = epochs
        print("Configuring perceptron with eta [{}], epochs [{}]".format(eta, epochs))

    def __threshold_array__(self, net_sums):
        print("net_sums: {}".format(net_sums))
        threshold_function = lambda x: 1 if x >= 0 else -1
        return np.array([threshold_function(x) for x in net_sums])

    def __threshold_number__(self, num):
        print("num: {}".format(num))
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
        print("Initial weights: {}".format(self.weights))
        for epoch in range(self.epochs):
            y_current = self.__threshold_array__(np.dot(self.weights, x_train_with_bias))
            self.weights = self.weights + np.dot(self.eta * (y_labels - y_current), x_train_with_bias.transpose())
            print("Epoch [{}], weights: {}".format(epoch, self.weights))

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


pn = Perceptron(eta=0.5, epochs=10)

x_train = np.array(
    [
        [0,0], [0, 1], # -1 samples
        [10,0], [10, 1], # 1 samples
    ]
)

y_lables = np.array([-1, -1, 1, 1])

pn.fit(x_train=x_train, y_labels=y_lables)

for row in range(x_train.shape[0]):
    predicted = pn.predict(x_train[row])
    print("Predicted [{}] for [{}]".format(predicted, x_train[row]))
