import matplotlib.pyplot as plt
from typing import List
import numpy as np


def plot_histogram_with_elite(scores: List[np.float], elite_score: np.float, epoch: int):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.hist(scores, bins=50)
    ax.set_title(f'elite score: {elite_score}')
    fig.savefig(f'plot_{epoch}.png')
    plt.close(fig)
