import numpy as np
import time
import tensorflow as tf
import concurrent
from ..genotype.individual import Individual
from ..evolution.sea import SEA
from ..environment.cart_pole import CartPole

np.random.seed(42)
tf.random.set_seed(42)


def xd(individuals):
    env = CartPole()
    return env.run_evaluation(individuals)


if __name__ == '__main__':

    LENGTH = 58

    # initial population
    pop = [Individual(LENGTH, [(4, 8), (8,), (8, 2), (2,)]) for i in range(100)]
    alg = SEA(0.8, 0.05)

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=10)

    for epoch in range(100):

        futures = []

        start = time.time()

        for individual in pop:
            fitness_future = executor.submit(xd, individual)
            futures.append(fitness_future)

        concurrent.futures.wait(futures)

        pop = list(map(lambda d: d.result(), futures))

        stop = time.time()

        print(stop - start)

        sorr = sorted(pop, key=lambda ind: -ind.last_fitness)

        promoted = sorr[:20]
        pop = [alg.mutate(promoted[i]) for i in np.random.randint(low=0, high=10, size=99)]
        pop += promoted[:1]

        print(f"best fitness for epoch {epoch} is {promoted[0].last_fitness}")

