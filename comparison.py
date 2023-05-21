import gym
import matplotlib.pyplot as plt

from algorithms.sarsa import Sarsa
from algorithms.sarsa_n import SarsaN
from algorithms.sarsa_lambda import SarsaLambda
from algorithms.qlearning import QLearning
from algorithms.expected_sarsa import ExpectedSarsa
from algorithms.dynaq import DynaQ

from utils.operating import run_repeatedly



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
        elif agent_params["algorithm"] == "sarsa_n":
            RLAgent = SarsaN
        elif agent_params["algorithm"] == "sarsa_lambda":
            RLAgent = SarsaLambda
        elif agent_params["algorithm"] == "qlearning":
            RLAgent = QLearning
        elif agent_params["algorithm"] == "expected_sarsa":
            RLAgent = ExpectedSarsa
        elif agent_params["algorithm"] == "dynaq":
            RLAgent = DynaQ
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
        plt.ylim([0,1500])
        plt.xlabel("episode")
        plt.ylabel("steps")
        plt.title("Avg. steps per episode")
        plt.legend(loc="upper right")
        plt.show()

    plot_learning_curve(alg_data, alg_labels)



if __name__ == '__main__':

    env = gym.make("MountainCar-v0")
    env._max_episode_steps = 3000

    params = [
        ("tab",
         {
             "algorithm" : "qlearning",
             "qfun_type" : "tabular",
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
             "plan_rep" : 50,
             "model_size" : 500,
         }),
        ("lin",
         {
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
             "plan_rep" : 50,
             "model_size" : 500,
         }),
        ("dynaq",
         {
             "algorithm" : "dynaq",
             "qfun_type" : "tabular",
             "bins" : [10,10],
             "coding_type" : "tile",
             "alpha" : 0.8,
             "gamma" : 1.0,
             "epsilon" : 0.1,
             "epsilon_decay" : 0.98,
             "min_epsilon" : 0.005,
             "n" : 5,
             "lambda" : 0.5,
             "et_type" : "accumulating",
             "plan_rep" : 50,
             "model_size" : 500,
         })
    ]

    algorithm_test(env, params[:2], num_repetitions=1,num_episodes=100)

    env.close()
