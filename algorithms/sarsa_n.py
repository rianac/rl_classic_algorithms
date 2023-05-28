from collections import deque
import gym.spaces as gsp

from utils.tabular_q_function import QValueFunction as TabularQ
from utils.linapprox_q_function import QValueFunction as LinearQ
from utils.exploration_policy import ExplorationPolicy



class SarsaN():
    def __init__(self, env, n, qfun_type, bins, coding_type,
                 alpha=0.1, gamma=0.99, **kwargs):

        assert isinstance(env.action_space, gsp.discrete.Discrete), \
            "Action space of environment is not discrete"
        assert isinstance(env.observation_space, gsp.box.Box), \
            "Observation space of environment is not continuous"
        assert (len(bins),) == env.observation_space.shape, \
            "Incompatible number of state dimensions"

        self.n = n

        self.alpha = alpha
        self.gamma = gamma

        self.bins = bins
        self.actions = list(range(env.action_space.n))

        self.policy = ExplorationPolicy(self.actions, **kwargs)

        self.prev_state = deque(maxlen=self.n)
        self.prev_action = deque(maxlen=self.n)
        self.rewards = deque(maxlen=self.n)

        feature_ranges = [[l,h] for l,h
                          in zip(list(env.observation_space.low),
                                 list(env.observation_space.high))]

        if qfun_type == "tabular":
            self.qfunction = TabularQ(feature_ranges, len(self.actions),
                                      self.bins, coding_type=coding_type)
        elif qfun_type == "linear_approx":
            self.qfunction = LinearQ(feature_ranges, len(self.actions),
                                     self.bins, coding_type=coding_type)
        else:
            unimplemented

    def reset(self):

        self.prev_state.extend([None]*self.n)
        self.prev_action.extend([None]*self.n)
        self.rewards.extend([None]*self.n)

        self.qfunction.reset()
        self.policy.reset()

    def reset_episode(self):

        self.prev_state.extend([None]*self.n)
        self.prev_action.extend([None]*self.n)
        self.rewards.extend([None]*self.n)

        self.policy.reset_episode()

    def act(self, reward, state, learning, done):

        self.rewards.append(reward)

        if not done:

            qvalues = [self.qfunction.value(state, action)
                       for action in self.actions]
            if learning:
                action = self.policy.get_action(qvalues)
            else:
                action = self.policy.best_action(qvalues)

            if self.prev_state[0] is not None and learning:
                self._learn(state, action, done)

            self.prev_state.append(state)
            self.prev_action.append(action)

            return action

        elif done and learning:
            while self.prev_state[0] is not None:
                self._learn(None, None, done)
                self.prev_state.append(None)
                self.prev_action.append(None)
                self.rewards.append(0)
        else:
            pass

    def _learn(self, state, action, done):

        rewards = 0
        for ind, rew in enumerate(self.rewards):
            rewards += self.gamma ** ind * rew

        if done:
            target = rewards
        else:
            q_state = self.qfunction.value(state, action)
            target = rewards + self.gamma ** self.n  * q_state

        self.qfunction.update(self.prev_state[0], self.prev_action[0],
                           target, self.alpha)


