import pandas as pd
import numpy as np
import net

PATH = "data/large-25x25.csv"
SIZE = 625
THRESHOLD = 0.5
ITERS = 10

def read_csv(path):
    try:
        with open(path, "r") as f:
            return pd.read_csv(f, header=None)
    except IOError as e:
        print(e)
        exit(1)

def add_noise(image, to_switch=10):
    noisy = image.copy()
    switch = np.random.choice(range(len(image)), to_switch, replace=False)
    noisy[switch] = -image[switch]
    return noisy

def get_random_image(size):
    img = np.random.random(size)
    img[img < 0.5] = -1
    img[img >= 0.5] = 1
    return img


#To print more elements in arrays
#np.set_printoptions(edgeitems=100)
data = read_csv(PATH).as_matrix()
print(data.shape)

network = net.HopfieldNet(SIZE, ITERS, THRESHOLD)
train = data
test = get_random_image(SIZE)#add_noise(data[5, :])
expected = data[5, :]
network.train(train)
result = network.test_async(test)
print("Input {},\n expected {}".format(test, expected))
print("Result", result)
print("Stable", (expected == result).all())
print("Same {} out of {}".format(sum(expected == result), len(expected)))
network.plot(expected, "Original")
