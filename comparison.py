from copy import deepcopy
import gym
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

from utils.operation_manager import run_repeatedly


from config import default_params



def algorithm_test(env, params, num_episodes=500, num_repetitions=10):
    """
    Testing performance of an agent (or more agents) during learning.
    The performance is measured by the number of steps taken to complete
    the task, averaged over a number of independent runs.
    """

    alg_labels = []
    alg_data = []

    for (label, agent_params) in params:

        if agent_params["algorithm"] == "sarsa":
            RLAgent = Sarsa
        elif agent_params["algorithm"] == "double_sarsa":
            RLAgent = DoubleSarsa
        elif agent_params["algorithm"] == "sarsa_n":
            RLAgent = SarsaN
        elif agent_params["algorithm"] == "sarsa_lambda":
            RLAgent = SarsaLambda
        elif agent_params["algorithm"] == "qlearning":
            RLAgent = QLearning
        elif agent_params["algorithm"] == "double_qlearning":
            RLAgent = DoubleQLearning
        elif agent_params["algorithm"] == "expected_sarsa":
            RLAgent = ExpectedSarsa
        elif agent_params["algorithm"] == "dynaq":
            RLAgent = DynaQ
        elif agent_params["algorithm"] == "osac":
            RLAgent = OneStepActorCritic
        else:
            unimplemented

        agent = RLAgent(env, **agent_params)
        data = run_repeatedly(env, agent, num_episodes, num_repetitions,
                              new_agent=True, training=True)
        alg_labels.append(label)
        alg_data.append(data)

    def plot_learning_curve(steps_per_episodes, algs):
        plt.figure(figsize=(10,5))
        for i in range(len(steps_per_episodes)):
            plt.plot(steps_per_episodes[i], label=algs[i])
        plt.ylim([0,550])
        plt.xlabel("episode")
        plt.ylabel("steps")
        plt.title("Avg. steps per episode")
        plt.legend(loc="upper right")
        plt.show()

    plot_learning_curve(alg_data, alg_labels)



if __name__ == '__main__':

    env = gym.make("MountainCar-v0")
    env._max_episode_steps = 600
    #env = gym.make("CartPole-v1")
    #env = gym.make("Acrobot-v1")


    params = [
        ["label1", deepcopy(default_params)],
        ["label2", deepcopy(default_params)],
        ["label3", deepcopy(default_params)],
        ["label4", deepcopy(default_params)]
    ]

    params[0][0] = "Sarsa"
    params[0][1]["algorithm"] = "sarsa"

    params[1][0] = "Double Sarsa"
    params[1][1]["algorithm"] = "double_sarsa"

    params[2][0] = "Q learning"
    params[2][1]["algorithm"] = "qlearning"

    params[3][0] = "Double Q learning"
    params[3][1]["algorithm"] = "double_qlearning"

    algorithm_test(env, params, num_repetitions=30, num_episodes=160)

    env.close()
