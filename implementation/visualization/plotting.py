import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rv_histogram
from numpy.random import Generator
from typing import List


def plot_median_with_intervals(elite_score_history, rng: Generator, out_dir):

    def compute_median_interval(data):
        med = np.median(data)
        medians = []
        for i in range(400):
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

    fig.savefig(f'{out_dir}/plot_median_intervals.png')
    plt.close(fig)

    np.save(f'{out_dir}/array_med.npy', meds)
    np.save(f'{out_dir}/array_high.npy', highs)
    np.save(f'{out_dir}/array_low.npy', lows)


def make_histogram(path: str, title: str, save_path: str):
    data = np.load(path)
    plt.hist(data, bins=50)
    plt.xlabel('score')
    plt.ylabel('frequency')
    plt.title(title)
    plt.savefig(save_path)


def make_plots(paths: List[str], legend: List[str], title: str, save_path: str,
               show_tops: bool = False, solved_at: float = None):
    def max_values(array):
        maxes = [array[0]]
        for i in array[1:]:
            if maxes[-1] < i:
                maxes.append(i)
            else:
                maxes.append(maxes[-1])
        return maxes

    fig, ax = plt.subplots()

    for path in paths:

        if path[-1] != '/':
            path += '/'

        meds = np.load(f'{path}array_med.npy')
        highs = np.load(f'{path}array_high.npy')
        lows = np.load(f'{path}array_low.npy')
        epochs = range(len(meds))

        if show_tops:
            meds = max_values(meds)
            highs = max_values(highs)
            lows = max_values(lows)

        ax.plot(epochs, meds)
        ax.fill_between(epochs, lows, highs, alpha=0.3)

    if solved_at is not None:
        ax.plot(epochs, [solved_at for _ in epochs], '--r')
        legend.insert(len(legend) // 2, 'solved at level')

    ax.legend(legend, loc='lower right')
    ax.set_xlabel('epoch')
    ax.set_ylabel('score')
    ax.set_title(title)
    fig.savefig(save_path)
