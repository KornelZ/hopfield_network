import numpy as np


class HopfieldNet(object):

    def __init__(self, size, iters, threshold):
        self.size = size
        self.W = np.zeros((size, size), dtype=np.float32)
        self.iters = iters
        self.threshold = np.full(size, fill_value=threshold, dtype=np.float32)

    def train(self, data):
        for i in range(data.shape[0]):
            self.W += np.outer(data[i, :], data[i, :])
        self.W /= len(data)
        np.fill_diagonal(self.W, 0)

    def sign(self, value):
        if value < 0:
            return -1
        return 1

    def sign_vec(self, value):
        return np.vectorize(self.sign)(value)

    def energy(self, input):
        return (-0.5 * np.sum(self.W * input.T * input) +
                np.sum(input * self.threshold))

    def test_async(self, input):
        output = input

        for iter in range(self.iters):
            order = list(range(self.size))
            np.random.shuffle(order)
            updated = False
            for i in order:
                activation = 0.0
                for j in range(self.size):
                    activation += self.W[i, j] * input[j]
                activation = self.sign(activation - self.threshold[0])
                if activation != output[i]:
                    output[i] = activation
                    updated = True
                    print("Updated")
            print("Iter {}, energy {}".format(iter, self.energy(output)))
            if not updated:
                break
        return output

    def test_sync(self, input):
        output = input
        for iter in range(self.iters):
            updated = False
            t = self.W * input
            activation = np.sum(self.W * input, axis=1)
            activation = self.sign_vec(activation - self.threshold)
            print("Iter {}, energy {}".format(iter, self.energy(output)))
            if (activation != output).any():
                output = activation
                updated = True
            if not updated:
                break
        return output

