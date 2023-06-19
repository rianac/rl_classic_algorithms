import random
import sys
from math import exp, log, isclose



class ExplorationPolicy():
    def __init__(self, actions, policy,
                 epsilon, epsilon_decay, min_epsilon, temperature, **kwargs):

        self.actions = actions
        self.nactions = len(self.actions)

        self.policy = policy    # "epsilon_greedy", "softmax", "max_boltzmann"

        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.epsilon = None

        self.temperature = temperature
        self.threshold = log(sys.float_info.max)

    def reset(self):

        if self.policy == "epsilon_greedy" or self.policy == "max_boltzmann":
            self.epsilon = self.initial_epsilon

    def reset_episode(self):

        if self.policy == "epsilon_greedy" or self.policy == "max_boltzmann":
            self.epsilon = max(self.epsilon * self.epsilon_decay,
                               self.min_epsilon)

    def get_distribution(self, qvalues):

        if self.policy == "epsilon_greedy":

            dist = [self.epsilon/self.nactions for _ in qvalues]

            idx = self.actions.index(self.best_action(qvalues))
            dist[idx] += 1 - self.epsilon

        elif self.policy == "softmax":

            max_qvalue = max(qvalues)
            if max_qvalue / self.temperature > self.threshold:
                temperature = max_qvalue / self.threshold + 0.001
            else:
                temperature = self.temperature

            dist = [exp(x/temperature) for x in qvalues]
            pow_sum = sum(dist)

            if isclose(pow_sum, 0.0, abs_tol=1e-300):
                dist = [1/self.nactions for _ in dist]
            else:
                dist = [x/pow_sum for x in dist]

        elif self.policy == "max_boltzmann":

            max_qvalue = max(qvalues)
            if max_qvalue / self.temperature > self.threshold:
                temperature = max_qvalue / self.threshold + 0.001
            else:
                temperature = self.temperature

            dist = [exp(x/temperature) for x in qvalues]
            pow_sum = sum(dist)

            if isclose(pow_sum, 0.0, abs_tol=1e-300):
                dist = [self.epsilon/self.nactions for _ in dist]
            else:
                dist = [self.epsilon*x/pow_sum for x in dist]

            idx = self.actions.index(self.best_action(qvalues))
            dist[idx] += 1 - self.epsilon

        else:
            unimplemented

        return dist

    def get_action(self, qvalues):

        probs = self.get_distribution(qvalues)
        [action] = random.choices(self.actions, probs)

        return action

    def best_action(self, qvalues):

        shuffled_actions = list(zip(self.actions[:], qvalues[:]))
        random.shuffle(shuffled_actions)

        best_action, maxq = shuffled_actions[0]
        for a, q in shuffled_actions:
            if q > maxq:
                best_action, maxq = a, q

        return best_action
