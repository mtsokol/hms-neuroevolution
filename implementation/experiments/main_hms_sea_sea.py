from ..evolution.hms.hms import HMS
from ..genotype.individual import Individual
from ..evolution.sea import SEA
from ..environment.gym_env import GymEnv
import concurrent
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

LENGTH = 58


def create_individual(level: int):
    return Individual(LENGTH, [(4, 8), (8,), (8, 2), (2,)], level)


def evaluate_individual(i):
    env = GymEnv(env_name="CartPole-v0")
    return env.run_evaluation(i)


if __name__ == '__main__':

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=10)

    hms = HMS(2, create_individual, evaluate_individual, [SEA(0.8, 0.5), SEA(0.8, 0.1)],
              [100, 50], [None, None], [5.1, 5.1], 200., 3, executor=executor)

    hms.run()

    print(hms.running_demes[0].elite.genotype_array)
    print(hms.running_demes[1].elite.genotype_array)
