import numpy as np


class LevelConfig:

    def __init__(self, mut_prob: np.float, mut_std: np.float, pop_size: int, promoted_num: int, lsc, spr_cond):
        self.mut_prob = mut_prob
        self.mut_std = mut_std
        self.pop_size = pop_size
        self.promoted_num = promoted_num
        self.lsc = lsc
        self.spr_cond = spr_cond
