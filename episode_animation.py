import gym

from algorithms.sarsa import Sarsa
from algorithms.double_sarsa import DoubleSarsa
from algorithms.sarsa_n import SarsaN
from algorithms.sarsa_lambda import SarsaLambda
from algorithms.qlearning import QLearning
from algorithms.double_qlearning import DoubleQLearning
from algorithms.expected_sarsa import ExpectedSarsa
from algorithms.dynaq import DynaQ
from algorithms.one_step_actor_critic import OneStepActorCritic

from utils.operation_manager import run_episodes

from config import default_params as params



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
    elif params["algorithm"] == "double_sarsa":
        RLAgent = DoubleSarsa
    elif params["algorithm"] == "sarsa_n":
        RLAgent = SarsaN
    elif params["algorithm"] == "sarsa_lambda":
        RLAgent = SarsaLambda
    elif params["algorithm"] == "qlearning":
        RLAgent = QLearning
    elif params["algorithm"] == "double_qlearning":
        RLAgent = DoubleQLearning
    elif params["algorithm"] == "expected_sarsa":
        RLAgent = ExpectedSarsa
    elif params["algorithm"] == "dynaq":
        RLAgent = DynaQ
    elif params["algorithm"] == "osac":
        RLAgent = OneStepActorCritic
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

    assert int(gym.__version__.split('.')[1]) <= 22, \
        "Monitor wrapper not present in gym version 23 and higher"

    env = gym.make("MountainCar-v0")
    env._max_episode_steps = 3000
    #env = gym.make("CartPole-v1")
    #env = gym.make("Acrobot-v1")


    animation_test(env,params, num_episodes=300)

    env.close()
