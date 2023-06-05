import numpy as np
import random

from codings.coding_selector import select_coding



class PiFunction():
    def __init__(self, env, actions, granularity, coding_type):

        self.actions = actions
        self.nactions = len(self.actions)

        self.action_probs = None
        self.action = None
        self.state_coding = None

        _, self.coding_size, self.discretizer = \
            select_coding(env, "linapprox", coding_type, granularity)

        self.weigths = None
        self.reset()

    def reset(self):
        self.weigths = np.zeros((self.nactions, self.coding_size))

    def get_action(self, state):
        self.state_coding = self.discretizer(state,vector_type=True)

        probs = np.exp(np.matmul(self.weigths, self.state_coding))
        self.action_probs = probs / sum(probs)

        [self.action] = random.choices(self.actions, self.action_probs)

        return self.action

    def update(self, td_error, alpha):

        features = self.state_coding.reshape((1,-1))
        probs = self.action_probs.reshape((-1,1))

        ln_gradient = -1 * probs * features
        ln_gradient[self.action] = (1 - probs[self.action]) * features

        self.weigths += alpha * td_error * ln_gradient

