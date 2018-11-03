import pandas as pd
import numpy as np
import net

PATH = "data/small-7x7.csv"
SIZE = 49
THRESHOLD = 0.5
ITERS = 10

def read_csv(path):
    try:
        with open(path, "r") as f:
            return pd.read_csv(f, header=None)
    except IOError as e:
        print(e)
        exit(1)

def add_noise(image, to_switch=4):
    noisy = image.copy()
    switch = np.random.choice(range(len(image)), to_switch)
    noisy[switch] = -image[switch]
    return noisy

#To print more elements in arrays
np.set_printoptions(edgeitems=100)
data = read_csv(PATH).as_matrix()
print(data.shape)

network = net.HopfieldNet(SIZE, ITERS, THRESHOLD)
train = data
test = add_noise(data[0, :])
expected = data[0, :]
network.train(train)
result = network.test_async(test)
print("Input {},\n expected {}".format(test, expected))
print("Result", result)
print("Stable", (expected == result).all())
print("Different {} out of {}".format(sum(expected == result), len(expected)))
