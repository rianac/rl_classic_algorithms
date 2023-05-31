
from functools import reduce
from operator import mul
from math import ceil, log

from codings.bin_coding import get_discretizer as get_bin_discretizer
from codings.tile_coding import get_discretizer as get_tile_discretizer
from codings.rbf_coding import get_discretizer as get_rbf_discretizer
from codings.polynomial_coding import get_discretizer as get_polynomial_discretizer
from codings.fourier_coding import get_discretizer as get_fourier_discretizer



def select_coding(env, representation, coding_type, bins):

    assert (len(bins),) == env.observation_space.shape, \
        "Incompatible number of state dimensions"

    if representation == "tabular":
        assert coding_type == "bin" or coding_type == "tile", \
            "Only bin / tile codings can be used for tabular representation"

    feature_ranges = [[l,h] for l,h
                      in zip(list(env.observation_space.low),
                             list(env.observation_space.high))]
    ntilings = None

    if coding_type == "bin":
        discretizer = get_bin_discretizer(feature_ranges, bins)
        ntilings = 1
        coding_size = reduce(mul,bins,1)
    elif coding_type == "tile":
        discretizer = get_tile_discretizer(feature_ranges, 8, bins)
        ntilings = 2 ** ceil(2 + log(len(bins), 2))
        coding_size = reduce(mul,bins,1) * ntilings
    elif coding_type == "rbf":
        discretizer = get_rbf_discretizer(feature_ranges, bins)
        coding_size = reduce(mul,bins,1)
    elif coding_type == "polynomial":
        discretizer = get_polynomial_discretizer(bins)
        coding_size = reduce(mul,bins,1)
    elif coding_type == "fourier":
        discretizer = get_fourier_discretizer(feature_ranges, bins)
        coding_size = reduce(mul,bins,1)
    elif coding_type == "fourier_simple":
        discretizer = get_fourier_discretizer(feature_ranges, bins, True)
        coding_size = sum(bins) - (len(bins) - 1)
    else:
        unimplemented

    return ntilings, coding_size, discretizer
