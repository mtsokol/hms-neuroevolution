import numpy as np
import multiprocessing as mp
import ctypes


def shared_noise_table() -> np.ndarray:
    seed = 123
    count = 250000000  # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
    print('Sampling {} random numbers with seed {}'.format(count, seed))
    shared_mem = mp.Array(ctypes.c_float, count, lock=False)
    noise = np.ctypeslib.as_array(shared_mem)
    assert noise.dtype == np.float32
    noise[:] = np.random.RandomState(seed).randn(count)  # 64-bit to 32-bit conversion here
    print('Sampled {} bytes'.format(noise.size * 4))
    return noise


if __name__ == '__name__':

    noise = shared_noise_table()
    np.save('noise.npy', noise)
