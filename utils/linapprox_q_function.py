import numpy as np
from functools import reduce
from operator import mul

from codings.bin_coding import get_discretizer as get_bin_discretizer
from codings.tile_coding import get_discretizer as get_tile_discretizer
from codings.rbf_coding import get_discretizer as get_rbf_discretizer
from codings.polynomial_coding import get_discretizer as get_polynomial_discretizer



class QValueFunction():
    def __init__(self, feature_ranges, num_actions, bins, coding_type,
                 lambda_val=None, et_type=None):

        self.et_type = et_type
        if self.et_type is not None:
            assert et_type == "accumulating", "Only accumulating ET possible"

            self.lambda_val = lambda_val
            self.zweigths = None

        self.num_actions = num_actions
        self.bins = reduce(mul,bins,1)

        self.coding_type = coding_type
        if self.coding_type == "bin":
            self.discretizer = get_bin_discretizer(feature_ranges, bins)
            self.num_tilings = 1
        elif self.coding_type == "tile":
            self.discretizer = get_tile_discretizer(feature_ranges, 8, bins)
            self.num_tilings = 8
        elif self.coding_type == "rbf":
            self.discretizer = get_rbf_discretizer(feature_ranges, bins)
            self.num_tilings = 1
        elif self.coding_type == "polynomial":
            self.discretizer = get_polynomial_discretizer(bins)
            self.num_tilings = 1
        else:
            unimplemented

        self.weigths = None
        self.reset()

    def reset(self):
        self.weigths = np.zeros((self.num_actions,
                                     self.bins * self.num_tilings))

    def reset_episode(self):
        if self.et_type is not None:
            self.zweigths = np.zeros(self.weigths[0].shape)

    def value(self, state, action, state_codings=None):
        if state_codings is None:
            state_codings = self.discretizer(state,vector_type=True)

        qvalue = self.weigths[action].dot(state_codings)

        return qvalue

    def update(self, state, action, target, alpha, gamma=None):
        state_codings = self.discretizer(state,vector_type=True)

        qvalue = self.value(state, action, state_codings)
        delta = alpha * (target - qvalue)

        if self.et_type is None:
            self.weigths[action] += state_codings * delta

        elif self.et_type == "accumulating":
            self.zweigths += state_codings
            self.weigths[action] += self.zweigths * delta
            self.zweigths *= gamma * self.lambda_val

        else:
            unimplemented


