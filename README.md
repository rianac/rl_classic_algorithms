# rl_classic_algorithms
Classic RL algorithms for Gymnasium environments with continuous state space and discrete action space

## Coding state space

Since the state space is continuous, states should be discretized into the form of a set of features.
The following state space coding methods are available:

- *aggregating coding* (selection `"aggregating"` or `"aggregating_simple"`) - dividing state space into separate bins
- *tile coding* (selection `"tile"` or `"tile_simple"`) - using several grids shifted towards each other
- *rbf coding* (selection `"rbf"` or `"rbf_simple"`) - defining a set of centers with subsequent measuring distances to these centers
- *fourier coding* (selection `"fourier"` or `"fourier_simple"`) - normalizing state space with subsequent use of *cos* functions

Selections `"*_simple"` treat each dimension in separation - multi-dimensional state space is considered as a combination of one-dimensional spaces. The other selections treat the state space in real multi-dimensional manner (but generate more features than the previous approach).  

Coding is controlled by `"granularity"` of the form of a list [n<sub>1</sub>, n<sub>2</sub>] where n<sub>i</sub> belongs to the i-th dimension of the state space and defines number of bins, centers or functions along the given dimension.

| method | type of features | # of features | # of features (simple)          |
|:-------:|:----------------:|:----:|:---:|
| aggregating | binary / indexes | n<sub>1</sub> * n<sub>2</sub> | n<sub>1</sub> + n<sub>2</sub> |
| tile | binary / indexes | n<sub>1</sub> * n<sub>2</sub> * #_of_tilings | (n<sub>1</sub> + n<sub>2</sub>) * #_of_tilings  |
| rbf  | real-valued | n<sub>1</sub> * n<sub>2</sub> | n<sub>1</sub> + n<sub>2</sub> |
| fourier | real-valued | n<sub>1</sub> * n<sub>2</sub>  | n<sub>1</sub> + n<sub>2</sub> - 1 |

Only coding methods producing binary features can be combined with tabular representation. Linear approximate representation can be combined with all provided coding methods.

## Exploration policy

Three types of exploration policies are available for algorithms learning action value function **Q** (algorithms for learning target policy **π** directly use this learned policy), :
- *ε-greedy* (selection `"epsilon_greedy"`) - with probability 1-ε is selected the action with the highest Q value, with probability ε is selected random action
- *softmax* (selection `"softmax"`) - actions are selected with probabilities given by their Q values 
- *max Boltzmann* (selection `"max_boltzmann"`) - with probability 1-ε is selected the action with the highest Q value, with probability ε are actions selected with probabilities given by their Q values

