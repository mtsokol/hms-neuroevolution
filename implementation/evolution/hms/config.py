import numpy as np
from typing import Optional, Tuple


class LevelConfig:

    def __init__(self, mut_prob: np.float, mut_std: np.float,
                 pop_size: int, promoted_num: int,
                 lsc: Optional[Tuple[str, float]], spr_cond: Optional[np.float]):
        self.mut_prob = mut_prob
        self.mut_std = mut_std
        self.pop_size = pop_size
        self.promoted_num = promoted_num
        self.lsc = lsc
        self.spr_cond = spr_cond

    def check_lsc(self, history: list, steps_run: int) -> bool:

        if self.lsc is None:
            return False

        lsc_type, value = self.lsc

        if lsc_type == 'epoch':
            return steps_run >= value
        elif lsc_type == 'obj_no_change':
            if len(history) > value:
                last_fitness = history[-1]
                for past_fitness in history[:-1][-value:]:
                    if last_fitness > past_fitness:
                        return False
                return True
            else:
                return False
        else:
            raise Exception('Invalid LSC')

    def __str__(self):
        return f'''Level config:
        \tmut_prob: {self.mut_prob} 
        \tmut_std: {self.mut_std} 
        \tpop_size: {self.pop_size}
        \tpromoted_num: {self.promoted_num}
        \tlsc: {self.lsc}
        \tspr_cond: {self.spr_cond}\n'''
