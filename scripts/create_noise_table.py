import numpy as np


if __name__ == '__main__':
    seed = 123
    count = 250000000
    print(f'Sampling {count} random numbers with seed {seed}')
    noise = np.random.RandomState(seed).randn(count).astype(np.float32)
    np.save('noise.npy', noise)
