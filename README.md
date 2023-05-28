# rl_classic_algorithms
Classic RL algorithms for Gymnasium environments with continuous state space and discrete action space

## Exploration policy

Three types of exploration policies are available:
- *ε-greedy* (selection `"epsilon_greedy"`) - with probability 1-ε is selected the action with the highest Q value, with probability ε is selected random action
- *softmax* (selection `"softmax"`) - actions are selected with probabilities given by their Q values 
- *max Boltzmann* (selection `"max_boltzmann"`) - with probability 1-ε is selected the action with the highest Q value, with probability ε are actions selected with probabilities given by their Q values

