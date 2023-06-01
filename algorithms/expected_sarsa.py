import gym.spaces as gsp

from utils.tabular_q_function import QValueFunction as TabularQ
from utils.linapprox_q_function import QValueFunction as LinearQ
from utils.exploration_policy import ExplorationPolicy



class ExpectedSarsa():
    def __init__(self, env, qfun_type, granularity, coding_type,
                 alpha=0.1, gamma=0.99, **kwargs):

        assert isinstance(env.action_space, gsp.discrete.Discrete), \
            "Action space of environment is not discrete"
        assert isinstance(env.observation_space, gsp.box.Box), \
            "Observation space of environment is not continuous"

        self.alpha = alpha
        self.gamma = gamma

        self.nactions = env.action_space.n
        self.actions = list(range(env.action_space.n))

        self.policy = ExplorationPolicy(self.actions, **kwargs)

        self.prev_state = None
        self.prev_action = None

        if qfun_type == "tabular":
            self.qfunction = TabularQ(env, len(self.actions),
                                      granularity, coding_type=coding_type)
        elif qfun_type == "linear_approx":
            self.qfunction = LinearQ(env, len(self.actions),
                                     granularity, coding_type=coding_type)
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
                self._learn(reward, state, done)

            self.prev_state = state
            self.prev_action = action

            return action

        elif done and learning and self.prev_state is not None:
            self._learn(reward, None, done)
        else:
            pass

    def _learn(self, reward, state, done):

        if done:
            target = reward
        else:
            qvalues = [self.qfunction.value(state, action)
                       for action in self.actions]

            probs = self.policy.get_distribution(qvalues)

            exp_value = 0
            for prob, qvalue in zip(probs, qvalues):
                exp_value += qvalue * prob

            target = reward + self.gamma * exp_value

        self.qfunction.update(self.prev_state, self.prev_action,
                           target, self.alpha)



