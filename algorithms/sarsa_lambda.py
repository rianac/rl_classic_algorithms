import gym.spaces as gsp

from utils.tabular_q_function import QValueFunction as TabularQ
from utils.linapprox_q_function import QValueFunction as LinearQ
from utils.exploration_policy import ExplorationPolicy



class SarsaLambda():
    def __init__(self, env, qfun_type, bins, coding_type,
                 alpha=0.1, gamma=0.99,
                 epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.1,
                 lambda_val=0.5, et_type="accumulating", **kwargs):

        assert isinstance(env.action_space, gsp.discrete.Discrete), \
            "Action space of environment is not discrete"
        assert isinstance(env.observation_space, gsp.box.Box), \
            "Observation space of environment is not continuous"
        assert (len(bins),) == env.observation_space.shape, \
            "Incompatible number of state dimensions"

        self.alpha = alpha
        self.gamma = gamma

        self.bins = bins
        self.actions = list(range(env.action_space.n))

        self.policy = ExplorationPolicy(self.actions,
                                        epsilon, epsilon_decay, min_epsilon)

        self.prev_state = None
        self.prev_action = None

        feature_ranges = [[l,h] for l,h
                          in zip(list(env.observation_space.low),
                                 list(env.observation_space.high))]

        if qfun_type == "tabular":
            self.qfunction = TabularQ(feature_ranges, len(self.actions),
                                      self.bins, coding_type=coding_type,
                                      lambda_val=lambda_val, et_type=et_type)
        elif qfun_type == "linear_approx":
            self.qfunction = LinearQ(feature_ranges, len(self.actions),
                                     self.bins, coding_type=coding_type,
                                     lambda_val=lambda_val, et_type=et_type)
        else:
            unimplemented

    def reset(self):

        self.prev_state = None
        self.prev_action = None

        self.qfunction.reset()
        self.policy.reset()

    def reset_episode(self):

        self.prev_state = None
        self.prev_action = None

        self.qfunction.reset_episode()
        self.policy.reset_episode()

    def act(self, reward, state, learning, done):

        if not done:

            qvalues = [self.qfunction.value(state, action)
                       for action in self.actions]
            if learning:
                action = self.policy.get_action(qvalues)
            else:
                action = self.policy.best_action(qvalues)

            if self.prev_state is not None and learning:
                self._learn(reward, state, action, done)

            self.prev_state = state
            self.prev_action = action

            return action

        elif done and learning and self.prev_state is not None:
            self._learn(reward, None, None, done)
        else:
            pass

    def _learn(self, reward, state, action, done):

        if done:
            target = reward
        else:
            q_state = self.qfunction.value(state, action)
            target = reward + self.gamma * q_state

        self.qfunction.update(self.prev_state, self.prev_action,
                              target, self.alpha, self.gamma)


