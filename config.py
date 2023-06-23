"""
Setting parameter values used by testing scripts as default parameter values.
"""



# Setting for exploration policy
#   - used by methods for learning q function only
exploration = {

    # policy types implemented: "epsilon_greedy", "softmax", "max_boltzmann"
    "policy" : "max_boltzmann",

    # linear decaying plan for epsilon_greedy/max_boltzmann policy
    "epsilon" : 1.0,
    "epsilon_decay" : 0.95,
    "min_epsilon" : 0.00001,

    # static plan for softmax/max_boltzmann policy
    "temperature" : 0.4,
}


# Setting for state space discretization
discretization = {

    # coding types implemented:
    #   "aggregating", "aggregating_simple", "tile", "tile_simple"
    #   "rbf", "rbf_simple", "fourier", "fourier_simple"
    "coding_type" : "tile",

    # number of dimensions depends on used model:
    #   Mountain Car : 2
    #   Cart Pole : 4
    #   Acrobot : 6
    "granularity" : [4,4,4,4,4,4],
}


# Setting for RL algorithm parameters
algorithm_params = {

    # algorithms implemented:
    #    "sarsa", "qlearning", "expected_sarsa"
    #    "sarsa_n", "sarsa_lambda"
    #    "osac" (one step actor critic), "dynaq"
    "algorithm" : "osac",

    # q function representation types:
    #    "tabular", "linear_approx"
    "qfun_type" : "linear_approx",

    # common learning parameters
    "alpha_w" : 0.01,
    "alpha_θ" : 0.01,
    "gamma" : 1.0,

    # sarsa_n specific parameters
    "n" : 5,

    # sarsa_lambda specific parameters
    "lambda_val" : 0.5,
    # et (eligibility trace) types:
    #   "accumulating" - for tabular as well as lin approx q fun representation
    #   "replacing"   - only for tabular q function representation
    "et_type" : "accumulating",

    # dynaq specific parameters
    "plan_rep" : 10,
    "model_size" : 500,
}


default_params = {}
default_params.update(algorithm_params)
default_params.update(discretization) 
default_params.update(exploration)
