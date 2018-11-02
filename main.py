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

#To print more elements in arrays
np.set_printoptions(edgeitems=100)
data = read_csv(PATH).as_matrix()
print(data.shape)

network = net.HopfieldNet(SIZE, ITERS, THRESHOLD)
train = data
test = data[0, :]
network.train(train)
result = network.test_sync(test)
print("Input", test)
print("Result", result)
print("Stable", (test == result).all())