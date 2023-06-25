import numpy as np
import gym
import matplotlib.colors as mplc
import matplotlib.pyplot as plt

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



def map_test(env, params,num_episodes = 500):
    """
    Drawing an action map for an agent learned over a specified number
    of episodes.

    State space is divided into tiles (defined by unique feature sets).
    Each tile is sampled by three samples in each dimension - if a tile
    represents a state subspace for which agent has not obtained any
    experience, then actions for these samples are selected randomly.

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

    feature_ranges = [[l,h] for l,h
                      in zip(list(env.observation_space.low),
                             list(env.observation_space.high))]

    num_tiles = [ 4 * x for x in params["granularity"]]

    x = np.linspace(feature_ranges[0][0],feature_ranges[0][1],6*num_tiles[0]+1)
    y = np.linspace(feature_ranges[1][0],feature_ranges[1][1],6*num_tiles[1]+1)
    x = x[1:-1:2]
    y = y[1:-1:2]

    xv, yv = np.meshgrid(x,y)
    zv = np.apply_along_axis(
        lambda obs: agent.act(None,obs,learning=False, done=False), 2,
        np.stack([xv, yv], axis=2))

    plt.figure(figsize=(8, 4))

    cmap = mplc.ListedColormap(["blue","white","green"])
    p = plt.pcolor(xv, yv, zv, cmap=cmap, vmin=-0.5, vmax=env.action_space.n-0.5)

    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title("Action map")
    plt.colorbar(p, shrink=0.7, ticks=[0,1,2])
    plt.show()



if __name__ == '__main__':

    env = gym.make("MountainCar-v0")
    env._max_episode_steps = 3000

    map_test(env,params, num_episodes=500)

    env.close()
