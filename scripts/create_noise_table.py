import numpy as np


if __name__ == '__name__':
    seed = 123
    count = 250000000
    noise = np.random.RandomState(seed).randn(count)
    np.save('noise.npy', noise)
