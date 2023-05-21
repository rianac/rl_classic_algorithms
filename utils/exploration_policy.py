import random
import numpy as np



class ExplorationPolicy():
    def __init__(self, actions, epsilon, epsilon_decay, min_epsilon):

        self.actions = actions

        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.epsilon = None

    def reset(self):

        self.epsilon = self.initial_epsilon

    def reset_episode(self):

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def get_epsilon(self):

        return self.epsilon

    def get_action(self, qvalues):

        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = self.best_action(qvalues)

        return action

    def best_action(self, qvalues):

        shuffled_actions = list(zip(self.actions[:], qvalues[:]))
        random.shuffle(shuffled_actions)

        actions = np.array(shuffled_actions)
        idx = np.argmax(actions, axis=0)[1]
        best_action = actions[idx,0]

        return int(best_action)

