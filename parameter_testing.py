import gym
import matplotlib.pyplot as plt

from algorithms.sarsa import Sarsa
from algorithms.double_sarsa import DoubleSarsa
from algorithms.sarsa_n import SarsaN
from algorithms.sarsa_lambda import SarsaLambda
from algorithms.true_sarsa_lambda import TrueSarsaLambda
from algorithms.qlearning import QLearning
from algorithms.double_qlearning import DoubleQLearning
from algorithms.expected_sarsa import ExpectedSarsa
from algorithms.dynaq import DynaQ
from algorithms.one_step_actor_critic import OneStepActorCritic

from utils.operation_manager import run_repeatedly

from config import default_params as params



def grid_test(env, default_params,
                   primary_param_name, primary_param_values,
                   secondary_param_name, secondary_param_values,
                   num_episodes=200, num_repetitions=10, skip_episodes=100):
    """
    Testing dependence of learning performance on values of two
    parameters (the other parameters keep their default values).

    The performance of each combination of parameters is measured
    by the number of steps taken to complete the task, averaged
    over a number of episodes and a number of independent runs.

    env : instance of RL environment
    default_params : values for required parameters
    primary_param_name : name of the first tested parameter
    primary_param_values : test values for the first tested parameter
    secondary_param_name : name of the second tested parameter
    secondary_param_values : test values for the second tested parameter
    num_episodes : number of episodes in one run
    num_repetitions : number of independent runs
    skip_episodes : number of skipped episodes at the beginning of
        each independent run
    """

    assert num_episodes > skip_episodes and skip_episodes >= 0, \
        "Suspicious number of episodes"
    assert num_repetitions > 0, "Nothing to average"

    # container for collecting test results
    data = []

    for p2 in secondary_param_values:
        default_params[secondary_param_name] = p2

        data_part = []
        for p1 in primary_param_values:
            default_params[primary_param_name] = p1

            if default_params["algorithm"] == "sarsa":
                RLAgent = Sarsa
            elif default_params["algorithm"] == "double_sarsa":
                RLAgent = DoubleSarsa
            elif default_params["algorithm"] == "sarsa_n":
                RLAgent = SarsaN
            elif default_params["algorithm"] == "sarsa_lambda":
                RLAgent = SarsaLambda
            elif default_params["algorithm"] == "true_sarsa_lambda":
                RLAgent = TrueSarsaLambda
            elif default_params["algorithm"] == "qlearning":
                RLAgent = QLearning
            elif default_params["algorithm"] == "double_qlearning":
                RLAgent = DoubleQLearning
            elif default_params["algorithm"] == "expected_sarsa":
                RLAgent = ExpectedSarsa
            elif default_params["algorithm"] == "dynaq":
                RLAgent = DynaQ
            elif default_params["algorithm"] == "osac":
                RLAgent = OneStepActorCritic
            else:
                unimplemented

            agent = RLAgent(env,**default_params)

            x = run_repeatedly(env, agent, num_episodes, num_repetitions,
                               new_agent=True, training=True)
            x = x[skip_episodes:]
            x = sum(x)/len(x)
            data_part.append(x)
        data.append(data_part)

    def plot_grid_search(x, xlabel, ys, ylabel, ylabels):
        x = [ str(y) for y in x]
        plt.figure(figsize=(5,5))
        for i in range(len(ylabels)):
            if ylabel == "other_params":
                plt.plot(x, ys[i], marker='o')
            else:
                plt.plot(x, ys[i], marker='o', label=ylabel+"="+str(ylabels[i]))
        plt.xlabel(xlabel)
        plt.ylabel("average number of steps")
        #plt.title("Grid search")
        plt.xticks(x)
        if ylabel != "other_params":
            plt.legend(loc="upper right")
        plt.show()

    plot_grid_search(primary_param_values, primary_param_name, data,
                     secondary_param_name, secondary_param_values)


def two_parameter_test(env, default_params,
                       primary_param_name, primary_param_values,
                       secondary_param_name, secondary_param_values,
                       **kwargs):
    """
    Testing dependence of learning performance on values of two
    parameters (the other parameters keep their default values).
    Based on grid testing of values of the two selected parameters.
    """
    grid_test(env,default_params,
              primary_param_name, primary_param_values,
              secondary_param_name, secondary_param_values,
              **kwargs)


def one_parameter_test(env, default_params,
                       primary_param_name, primary_param_values,
                       **kwargs):
    """
    Testing dependence of learning performance on values of one
    parameter (the other parameters keep their default values).
    Based on grid testing of the selected parameter and one
    dummy parameter.
    """

    dummy_param_name = "other_params"
    dummy_param_values = ["defaults"]

    grid_test(env,default_params,
              primary_param_name, primary_param_values,
              dummy_param_name, dummy_param_values,
              **kwargs)



if __name__ == '__main__':

    #env = gym.make("MountainCar-v0")
    #env._max_episode_steps = 500
    #env = gym.make("CartPole-v1")
    env = gym.make("Acrobot-v1")


    tested_param1 = "algorithm"
    tested_param1_values = ["sarsa_n","sarsa_lambda", "dynaq", "osac"]

    tested_param2 = None
    tested_param2_values = []

    if tested_param2 is not None:
        two_parameter_test(env, params, 
                           tested_param1, tested_param1_values,
                           tested_param2, tested_param2_values,
                           num_repetitions=5,
                           skip_episodes=100, num_episodes=150)
    else:
        one_parameter_test(env, params, 
                           tested_param1, tested_param1_values,
                           num_repetitions=5, 
                           skip_episodes=100, num_episodes=150)

    env.close()
