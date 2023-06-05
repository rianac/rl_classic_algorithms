import random
from collections import deque
import gym.spaces as gsp

from utils.tabular_q_function import QValueFunction as TabularQ
from utils.linapprox_q_function import QValueFunction as LinearQ
from utils.exploration_policy import ExplorationPolicy


class DynaQ():
    def __init__(self, env, qfun_type, granularity, coding_type,
                 alpha_w, gamma, plan_rep, model_size,
                 **kwargs):

        assert isinstance(env.action_space, gsp.discrete.Discrete), \
            "Action space of environment is not discrete"
        assert isinstance(env.observation_space, gsp.box.Box), \
            "Observation space of environment is not continuous"

        self.alpha = alpha_w
        self.gamma = gamma

        self.plan_rep = plan_rep
        self.model_size = model_size

        self.actions = list(range(env.action_space.n))

        self.policy = ExplorationPolicy(self.actions, **kwargs)

        self.prev_state = None
        self.prev_action = None

        self.model = None

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
        self.model = deque(maxlen=self.model_size)

    def reset_episode(self):

        self.prev_state = None
        self.prev_action = None

        self.policy.reset_episode()

    def act(self, reward, state, learning, done):

        if self.prev_state is not None:
            self.model.append((self.prev_state, self.prev_action,
                               reward, state, done))

        if not done:

            qvalues = [self.qfunction.value(state, action)
                       for action in self.actions]
            if learning:
                action = self.policy.get_action(qvalues)
            else:
                action = self.policy.best_action(qvalues)

            if self.prev_state is not None and learning:
                self._learn(self.prev_state, self.prev_action,
                            reward, state, done)
                self._plan()

            self.prev_state = state
            self.prev_action = action

            return action

        elif done and learning and self.prev_state is not None:
            self._learn(self.prev_state, self.prev_action,
                        reward, None, done)
            self._plan()
        else:
            pass

    def _learn(self, prev_state, prev_action, reward, state, done):

        if done:
            target = reward
        else:
            qvalues = [self.qfunction.value(state, action)
                       for action in self.actions]
            best_action = self.policy.best_action(qvalues)
            max_q_value = qvalues[best_action]
            target = reward + self.gamma * max_q_value

        self.qfunction.update(prev_state, prev_action,
                           target, self.alpha)

    def _plan(self):

        for _ in range(self.plan_rep):
            state, action, reward, next_state, done = random.choice(self.model)

            self._learn(state, action, reward, next_state, done)
