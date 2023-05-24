import numpy as np
import gym
import matplotlib.colors as mplc
import matplotlib.pyplot as plt

from algorithms.sarsa import Sarsa
from algorithms.sarsa_n import SarsaN
from algorithms.sarsa_lambda import SarsaLambda
from algorithms.qlearning import QLearning
from algorithms.expected_sarsa import ExpectedSarsa
from algorithms.dynaq import DynaQ

from utils.operation_manager import run_episodes



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

    feature_ranges = [[l,h] for l,h
                      in zip(list(env.observation_space.low),
                             list(env.observation_space.high))]

    if params["coding_type"] == "bin":
        num_tiles = params["bins"]
    elif params["coding_type"] == "tile":
        num_tiles = [ (x-1)*8 + 1 for x in params["bins"]]
    elif params["coding_type"] == "rbf":
        num_tiles = params["bins"]
    elif params["coding_type"] == "polynomial":
        num_tiles = params["bins"]
    else:
        unimplemented

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
    plt.title("Greedy policy based on q-value function")
    plt.colorbar(p, shrink=0.7, ticks=[0,1,2])
    plt.show()



if __name__ == '__main__':

    env = gym.make("MountainCar-v0")
    env._max_episode_steps = 3000

    params = {
        "algorithm" : "sarsa",
        "qfun_type" : "linear_approx",
        "bins" : [10,10],
        "coding_type" : "rbf",
        "alpha" : 0.1,
        "gamma" : 1.0,
        "epsilon" : 0.1,
        "epsilon_decay" : 0.98,
        "min_epsilon" : 0.005,
        "n" : 5,
        "lambda_val" : 0.5,
        "et_type" : "accumulating",
        "plan_rep" : 9,
        "model_size" : 500,
    }

    map_test(env,params,num_episodes=100)

    env.close()
