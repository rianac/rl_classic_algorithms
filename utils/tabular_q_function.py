import numpy as np
from functools import reduce
from itertools import product

from codings.bin_coding import get_discretizer as get_bin_discretizer
from codings.tile_coding import get_discretizer as get_tile_discretizer



class QValueFunction():
    def __init__(self, feature_ranges, num_actions, bins, coding_type,
                 lambda_val=None, et_type=None):

        self.et_type = et_type
        if self.et_type is not None:
            self.lambda_val = lambda_val
            self.etables = None

        self.num_actions = num_actions
        self.bins = bins

        if coding_type == "bin":
            self.discretizer = get_bin_discretizer(feature_ranges, bins)
            self.num_tilings = 1
        elif coding_type == "tile":
            self.discretizer = get_tile_discretizer(feature_ranges, 8, bins)
            self.num_tilings = 8
        else:
            unimplemented

        self.qtables = None
        self.reset()

    def reset(self):
        self.qtables = [np.zeros(self.bins + [self.num_actions])
                         for _ in range(self.num_tilings)]

    def reset_episode(self):
        if self.et_type is not None:
            self.etables = [np.zeros(self.bins + [self.num_actions])
                            for _ in range(self.num_tilings)]

    def value(self, state, action):
        state_codings = self.discretizer(state,vector_type=False)

        qvalue = [qtable[coding + (action,)]
                  for qtable, coding in zip(self.qtables, state_codings)]
        qvalue = sum(qvalue) / self.num_tilings

        return qvalue

    def update(self, state, action, target, alpha, gamma=None):
        state_codings = self.discretizer(state, vector_type=False)

        if self.et_type is None:
            for qtable, coding in zip(self.qtables, state_codings):
                delta = target - qtable[coding + (action,)]
                qtable[coding + (action,)] += alpha * delta

        elif self.et_type is not None:
            for qtable, coding, etable in zip(self.qtables,
                                              state_codings,
                                              self.etables):

                delta = target - qtable[coding + (action,)]

                if self.et_type == "accumulating":
                    etable[coding + (action,)] += 1
                elif self.et_type == "replacing":
                    etable[coding + (action,)] = 1
                else:
                    unimplemented

                qtable += alpha * delta * etable
                etable *= gamma * self.lambda_val

