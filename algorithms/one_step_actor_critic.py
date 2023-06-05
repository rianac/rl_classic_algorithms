import gym.spaces as gsp

from utils.linapprox_v_function import VValueFunction as VFun
from utils.linapprox_policy import PiFunction as Policy



class OneStepActorCritic():
    def __init__(self, env, granularity, coding_type, gamma,
                 alpha_w, alpha_θ, **kwargs):

        assert isinstance(env.action_space, gsp.discrete.Discrete), \
            "Action space of environment is not discrete"
        assert isinstance(env.observation_space, gsp.box.Box), \
            "Observation space of environment is not continuous"
        assert (len(granularity),) == env.observation_space.shape, \
            "Incompatible number of state dimensions"

        self.alpha_w = alpha_w
        self.alpha_θ = alpha_θ
        self.gamma = gamma

        self.actions = list(range(env.action_space.n))
        self.prev_state = None

        self.policy = Policy(env, self.actions,
                             granularity, coding_type=coding_type)

        self.vfunction = VFun(env, granularity, coding_type=coding_type)

    def reset(self):

        self.prev_state = None

        self.vfunction.reset()
        self.policy.reset()

    def reset_episode(self):

        self.prev_state = None

    def act(self, reward, state, learning, done):

        if not done:

            if self.prev_state is not None and learning:
                self._learn(reward, state, done)

            action = self.policy.get_action(state)

            self.prev_state = state

            return action

        elif done and learning and self.prev_state is not None:
            self._learn(reward, None, done)
        else:
            pass

    def _learn(self, reward, state, done):

        if done:
            target = reward
        else:
            next_v_value = self.vfunction.value(state)
            target = reward + self.gamma * next_v_value

        v_value = self.vfunction.value(self.prev_state)
        td_error = target - v_value

        self.vfunction.update(self.prev_state, td_error, self.alpha_w)
        self.policy.update(td_error, self.alpha_θ)
