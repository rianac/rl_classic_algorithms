import gym

from algorithms.sarsa import Sarsa
from algorithms.sarsa_n import SarsaN
from algorithms.sarsa_lambda import SarsaLambda
from algorithms.qlearning import QLearning
from algorithms.expected_sarsa import ExpectedSarsa
from algorithms.dynaq import DynaQ

from utils.operation_manager import run_episodes



def animation_test(env, params, num_episodes=500):
    """
    Testing an agent (learned over a specified number of episodes)
    on solving episode.

    env : instance of RL environment
    params : values for required parameters
    num_episodes : number episodes used for training
    """

    if params["algorithm"] == "sarsa":
        RLAgent = Sarsa
    elif params["algorithm"] == "sarsa_n":
        RLAgent = SarsaN
    elif params["algorithm"] == "sarsa_lambda":
        RLAgent = SarsaLambda
    elif params["algorithm"] == "qlearning":
        RLAgent = QLearning
    elif params["algorithm"] == "expected_sarsa":
        RLAgent = ExpectedSarsa
    elif params["algorithm"] == "dynaq":
        RLAgent = DynaQ
    else:
        unimplemented

    agent = RLAgent(env, **params)
    run_episodes(env, agent, num_episodes, new_agent=True, training=True)

    menv = gym.wrappers.Monitor(env, './video/', force=True)

    state = menv.reset()

    num_steps = 0
    done = False
    while not done:
        action = agent.act(None, state, learning=False, done=done)
        state, _, done, _ = menv.step(action)
        num_steps += 1
        menv.render()

    print('Solved in {} steps'.format(num_steps))

    menv.close()


if __name__ == '__main__':

    env = gym.make("MountainCar-v0")
    env._max_episode_steps = 3000

    params = {
        "algorithm" : "qlearning",
        "qfun_type" : "linear_approx",
        "bins" : [10,10],
        "coding_type" : "tile",
        "alpha" : 0.1,
        "gamma" : 1.0,
        "epsilon" : 0.1,
        "epsilon_decay" : 0.98,
        "min_epsilon" : 0.005,
        "n" : 5,
        "lambda" : 0.5,
        "et_type" : "accumulating",
        "plan_rep" : 10,
        "model_size" : 500,
    }

    animation_test(env,params,num_episodes=200)

    env.close()
