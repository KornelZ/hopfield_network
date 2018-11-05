import pandas as pd
import numpy as np
import net

PATH = "data/large-25x25.csv"
IMG_SIZE = (25, 25)
SIZE = IMG_SIZE[0] * IMG_SIZE[1]
THRESHOLD = 0.5
ITERS = 10
ACCEPTABLE_PERCENT = 1.0
TEST_SYNC = True

USE_RANDOM_IMAGE = False
NOISE = 30
SAVE_PLOTS = False


def read_csv(path):
    try:
        with open(path, "r") as f:
            return pd.read_csv(f, header=None)
    except IOError as e:
        print(e)
        exit(1)

def add_noise(image, to_switch=NOISE):
    noisy = image.copy()
    switch = np.random.choice(range(len(image)), to_switch, replace=False)
    noisy[switch] = -image[switch]
    return noisy

def get_random_image(size):
    img = np.random.random(size)
    img[img < 0.5] = -1
    img[img >= 0.5] = 1
    return img

def test_dataset(data):
    network = net.HopfieldNet(SIZE, ITERS, THRESHOLD, SAVE_PLOTS, IMG_SIZE)
    train = data
    network.train(train)
    total_stable = 0
    for i in range(data.shape[0]):
        print("PATTERN #", i + 1)
        test = add_noise(data[i, :])
        expected = data[i, :]
        result = network.test(test, TEST_SYNC)
        stability = expected == result
        total = len(stability)
        is_stable = sum(stability) >= ACCEPTABLE_PERCENT * total
        print("Stable", is_stable)
        print("Same {} out of {}".format(sum(stability), len(expected)))
        network.plot(expected, "Original")
        if is_stable:
            total_stable += 1
    print("Total stable ", total_stable)


def test_random(data):
    network = net.HopfieldNet(SIZE, ITERS, THRESHOLD, SAVE_PLOTS, IMG_SIZE)
    train = data
    network.train(train)
    test = get_random_image(SIZE)
    result = network.test(test, TEST_SYNC)
    print("Random result:", result)


#To print more elements in arrays
np.set_printoptions(edgeitems=10)
data = read_csv(PATH).as_matrix()

if USE_RANDOM_IMAGE:
    test_random(data)
else:
    test_dataset(data)

