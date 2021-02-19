from ..evolution.hms.hms import HMS
from ..genotype.individual import Individual
from ..evolution.sea import SEA
from ..environment.gym_env import GymEnv
import concurrent

LENGTH = 18180


def create_individual(level: int):
    return Individual(LENGTH, [(8, 128), (128,), (128, 128), (128,), (128, 4), (4,)], level)


def evaluate_individual(i):
    env = GymEnv(env_name='LunarLander-v2')
    return env.run_evaluation(i)


if __name__ == '__main__':

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=10)

    hms = HMS(1, create_individual, evaluate_individual, [SEA(0.8, 0.5)],
              [200], [None], [5.1], 200., 10, executor=executor)

    hms.run()
