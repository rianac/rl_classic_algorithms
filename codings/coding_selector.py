from functools import reduce
from operator import mul
from math import ceil, log

from codings.aggregating_coding import get_discretizer as get_aggregating_discretizer
from codings.tile_coding import get_discretizer as get_tile_discretizer
from codings.rbf_coding import get_discretizer as get_rbf_discretizer
from codings.fourier_coding import get_discretizer as get_fourier_discretizer



def select_coding(env, representation, coding_type, granularity):

    assert (len(granularity),) == env.observation_space.shape, \
        "Incompatible number of state dimensions"

    if representation == "tabular":
        assert coding_type == "aggregating" or coding_type == "tile", \
            "Only bin / tile codings can be used for tabular representation"

    if type(env.unwrapped).__name__  == 'MountainCarEnv':
        feature_ranges = [[l,h] for l,h
                          in zip(list(env.observation_space.low),
                                 list(env.observation_space.high))]
    elif type(env.unwrapped).__name__  == 'CartPoleEnv':
        feature_ranges = [
            [-2.5, 2.5],     # cart position   -4.8, 4.8
            [-3.5, 3.5],     # cart velocity   -inf, inf
            [-0.28, 0.28],   # pole angle      -0.418, 0.418
            [-3.8, 3.8]      # pole velocity   -inf, inf
        ]
    elif type(env.unwrapped).__name__  == 'AcrobotEnv':
        feature_ranges = [[l,h] for l,h
                          in zip(list(env.observation_space.low),
                                 list(env.observation_space.high))]
    else:
        unimplemented

    ntilings = None

    if coding_type == "aggregating":
        discretizer = get_aggregating_discretizer(feature_ranges, granularity)
        ntilings = 1
        coding_size = reduce(mul,granularity,1)

    elif coding_type == "aggregating_simple":
        discretizer = get_aggregating_discretizer(feature_ranges, granularity,
                                                  True)
        ntilings = 1
        coding_size = sum(granularity)

    elif coding_type == "tile":
        ntilings = 2 ** ceil(2 + log(len(granularity), 2))
        discretizer = get_tile_discretizer(feature_ranges, ntilings,
                                           granularity)
        coding_size = reduce(mul,granularity,1) * ntilings

    elif coding_type == "tile_simple":
        ntilings = 4
        discretizer = get_tile_discretizer(feature_ranges, ntilings,
                                           granularity, True)
        coding_size = sum(granularity) * ntilings

    elif coding_type == "rbf":
        discretizer = get_rbf_discretizer(feature_ranges, granularity)
        coding_size = reduce(mul,granularity,1)

    elif coding_type == "rbf_simple":
        discretizer = get_rbf_discretizer(feature_ranges, granularity, True)
        coding_size = sum(granularity)

    elif coding_type == "fourier":
        discretizer = get_fourier_discretizer(feature_ranges, granularity)
        coding_size = reduce(mul,granularity,1)

    elif coding_type == "fourier_simple":
        discretizer = get_fourier_discretizer(feature_ranges, granularity, True)
        coding_size = sum(granularity) - (len(granularity) - 1)

    else:
        unimplemented

    return ntilings, coding_size, discretizer
