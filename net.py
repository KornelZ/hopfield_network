import numpy as np
import matplotlib.pyplot as plt
import time
import os

def save_plot():
    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    index = ""
    while os.path.isfile(timestr + str(index) + ".png"):
        if index == "":
            index = 1
        else:
            index += 1
    plt.savefig(timestr + str(index) + ".png")


class HopfieldNet(object):

    def __init__(self, size, iters, threshold, save_plots, img_size):
        self.size = size
        self.W = np.zeros((size, size), dtype=np.float32)
        self.iters = iters
        self.threshold = np.full(size, fill_value=threshold, dtype=np.float32)
        self.save_plots = save_plots
        self.img_size = img_size

    def train(self, data):
        for i in range(data.shape[0]):
            self.W += np.outer(data[i, :], data[i, :])
        self.W /= len(data)
        np.fill_diagonal(self.W, 0)

    def sign(self, value):
        value = np.sign(value)
        value[value == 0] = 1
        return value

    def sign_vec(self, value):
        return np.vectorize(self.sign)(value)

    def energy(self, input):
        return (-0.5 * np.dot(np.dot(input.T, self.W), input) +
                np.sum(input * self.threshold))

    def test_async(self, input):
        output = input
        for iter in range(self.iters):
            order = list(range(self.size))
            np.random.shuffle(order)
            updated = False
            print("Iter {}, energy {}".format(iter, self.energy(output)))
            self.plot(output, iter)
            for i in order:
                activation = np.dot(self.W[i, :],  input)
                activation = np.sign(activation - self.threshold[0])
                if activation != output[i]:
                    output[i] = activation
                    updated = True
            if not updated:
                break

        return output

    def test_sync(self, input):
        output = input
        for iter in range(self.iters):
            updated = False
            activation = np.dot(self.W, input)
            activation = self.sign(activation - self.threshold)
            print("Iter {}, energy {}".format(iter, self.energy(output)))
            self.plot(output, iter)
            if (activation != output).any():
                output = activation
                updated = True
                print("Updated")
            if not updated:
                break

        return output

    def test(self, input, sync):
        if sync:
            return self.test_sync(input)
        else:
            return self.test_async(input)

    def plot(self, output, title):
        if isinstance(title, int):
            plt.title("Iter {}".format(title))
        else:
            plt.title(title)
        img = np.resize(output, self.img_size)

        plt.imshow(img)
        if self.save_plots:
            save_plot()
        plt.show()

