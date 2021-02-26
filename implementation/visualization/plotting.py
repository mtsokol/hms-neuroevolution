import matplotlib.pyplot as plt
from typing import List
import numpy as np
from uuid import UUID
from scipy.stats import rv_histogram
from numpy.random import Generator


def plot_histogram_with_elite(scores: List[np.float], elite_score: np.float, epoch: int, deme_id: UUID):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.hist(scores, bins=50)
    ax.set_title(f'elite score: {elite_score}')
    fig.savefig(f'plot_{epoch}_deme_{deme_id}.png')
    plt.close(fig)


def plot_median_with_intervals(elite_score_history, rng: Generator):

    def compute_median_interval(data):
        med = np.median(data)
        medians = []
        for i in range(1000):
            choice = rng.choice(data, len(data))
            medians.append(np.median(choice))
        rv = rv_histogram(np.histogram(medians))
        low, high = rv.interval(0.95)
        return med, low, high

    epochs = range(len(elite_score_history))
    meds = []
    lows = []
    highs = []

    for elite_score in elite_score_history:
        med, low, high = compute_median_interval(elite_score)
        meds.append(med), lows.append(low), highs.append(high)

    plt.style.use('seaborn')
    fig, ax = plt.subplots()

    ax.plot(epochs, meds)
    ax.fill_between(epochs, lows, highs, alpha=0.3)

    fig.savefig(f'plot_median_intervals.png')
    plt.close(fig)
