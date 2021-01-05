from ..evolution.hms.hms import HMS
from ..genotype.individual import Individual
from ..evolution.sea import SEA
from ..environment.cart_pole import CartPole
import concurrent

LENGTH = 58


def create_individual(level: int):
    return Individual(LENGTH, [(4, 8), (8,), (8, 2), (2,)], level)


def evaluate_individual(i):
    env = CartPole()
    return env.run_evaluation(i)


if __name__ == '__main__':

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=10)

    hms = HMS(1, create_individual, evaluate_individual, [SEA(0.8, 0.1)],
              [100], [None], [5.1], 200., 10, executor=executor)

    hms.run()
