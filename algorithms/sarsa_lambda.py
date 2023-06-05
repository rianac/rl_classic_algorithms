import gym.spaces as gsp

from utils.tabular_q_function import QValueFunction as TabularQ
from utils.linapprox_q_function import QValueFunction as LinearQ
from utils.exploration_policy import ExplorationPolicy



class SarsaLambda():
    def __init__(self, env, qfun_type, granularity, coding_type,
                 alpha_w, gamma,
                 lambda_val=0.5, et_type="accumulating", **kwargs):

        assert isinstance(env.action_space, gsp.discrete.Discrete), \
            "Action space of environment is not discrete"
        assert isinstance(env.observation_space, gsp.box.Box), \
            "Observation space of environment is not continuous"

        self.alpha = alpha_w
        self.gamma = gamma

        self.actions = list(range(env.action_space.n))

        self.policy = ExplorationPolicy(self.actions, **kwargs)

        self.prev_state = None
        self.prev_action = None

        if qfun_type == "tabular":
            self.qfunction = TabularQ(env, len(self.actions),
                                      granularity, coding_type=coding_type,
                                      lambda_val=lambda_val, et_type=et_type)
        elif qfun_type == "linear_approx":
            self.qfunction = LinearQ(env, len(self.actions),
                                     granularity, coding_type=coding_type,
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


