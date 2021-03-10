import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rv_histogram
from numpy.random import Generator


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
