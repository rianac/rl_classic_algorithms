import numpy as np
import random

from codings.coding_selector import select_coding



class PiFunction():
    def __init__(self, env, actions, granularity, coding_type):

        self.actions = actions
        self.nactions = len(self.actions)

        _, self.coding_size, self.discretizer = \
            select_coding(env, "linapprox", coding_type, granularity)

        self.weigths = None
        self.reset()

    def reset(self):
        self.weigths = np.zeros((self.nactions, self.coding_size))

    def _get_action_probs(self, state_coding):
        probs = np.exp(np.matmul(self.weigths, state_coding))
        action_probs = probs / sum(probs)

        return action_probs
        
    def get_action(self, state):
        state_coding = self.discretizer(state,vector_type=True)
        
        action_probs = self._get_action_probs(state_coding)       
        [action] = random.choices(self.actions, action_probs)

        return action

    def update(self, state, action, td_error, alpha):
        state_coding = self.discretizer(state,vector_type=True)        
        features = state_coding.reshape((1,-1))

        probs = self._get_action_probs(state_coding).reshape((-1,1))
        
        ln_gradient = -1 * probs * features
        ln_gradient[action] = (1 - probs[action]) * features

        self.weigths += alpha * td_error * ln_gradient

