import numpy as np

from codings.coding_selector import select_coding



class VValueFunction():
    def __init__(self, env, granularity, coding_type):

        _, self.coding_size, self.discretizer = \
            select_coding(env, "linapprox", coding_type, granularity)

        self.weigths = None
        self.reset()

    def reset(self):
        self.weigths = np.zeros((self.coding_size,))

    def value(self, state):
        state_codings = self.discretizer(state,vector_type=True)

        v_value = self.weigths.dot(state_codings)

        return v_value

    def update(self, state, td_error, alpha):
        state_codings = self.discretizer(state,vector_type=True)

        self.weigths += state_codings * alpha * td_error
