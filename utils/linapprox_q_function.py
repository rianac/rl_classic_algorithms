import numpy as np

from codings.coding_selector import select_coding



class QValueFunction():
    def __init__(self, env, num_actions, granularity, coding_type,
                 lambda_val=None, et_type=None):

        self.et_type = et_type
        if self.et_type is not None:
            assert et_type == "accumulating" or et_type == "dutch", \
              "Only accumulating or dutch ET possible"

            self.lambda_val = lambda_val
            self.zweigths = None

        self.num_actions = num_actions

        _, self.coding_size, self.discretizer = \
            select_coding(env, "linapprox", coding_type, granularity)

        self.weigths = None
        self.reset()

    def reset(self):
        self.weigths = np.zeros((self.num_actions, self.coding_size))

    def reset_episode(self):
        if self.et_type is not None:
            self.zweigths = np.zeros(self.weigths[0].shape)

    def value(self, state, action, state_codings=None):
        if state_codings is None:
            state_codings = self.discretizer(state,vector_type=True)

        qvalue = self.weigths[action].dot(state_codings)

        return qvalue

    def update(self, state, action, target, alpha, gamma=None, q_old=None):
        state_codings = self.discretizer(state,vector_type=True)

        qvalue = self.value(state, action, state_codings)
        delta = alpha * (target - qvalue)

        if self.et_type is None:
            self.weigths[action] += state_codings * delta

        elif self.et_type == "accumulating":
            self.zweigths += state_codings
            self.weigths[action] += self.zweigths * delta
            self.zweigths *= gamma * self.lambda_val

        elif self.et_type == "dutch":
            self.zweigths += state_codings * \
              (1.0 - alpha * gamma * self.lambda_val * \
                     self.zweigths.dot(state_codings))
            self.weigths[action] += self.zweigths * delta + \
               alpha * (qvalue - q_old) * (self.zweigths - state_codings)
            self.zweigths *= gamma * self.lambda_val

        else:
            unimplemented


