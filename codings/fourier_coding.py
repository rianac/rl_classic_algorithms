"""
Fourier-base based coding

"""

import numpy as np
from itertools import product



def get_discretizer(feature_ranges, orders, simple=False):
    """
    feature_ranges: range of each feature
        example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
    number_orderss: bin size for each dimension
        example: 8 orders for x and 6 orders for y -> [8, 6]

    return: fourier cos base coder
    """

    num_dims = len(feature_ranges)
    assert num_dims == len(orders), "Dimensionality mismatch"

    # Find normalisation constants for (x - n1) / n2
    n1 = np.array([feat_range[0] for feat_range in feature_ranges])
    n2 = np.array([feat_range[1] - feat_range[0]
                   for feat_range in feature_ranges])

    # Find dimension multiplies
    tmp = [range(order) for order in orders]
    coefs = np.transpose(np.array([comb for comb in product(*tmp)]))

    if simple:
        selections = np.sum(coefs.clip(0,1),axis=0) <= 1
        coefs = coefs[:,selections]

    def discretizer(features, **kwargs):
        """
        feature: sample with multiple dimensions to be encoded;
            example: x = 0.8 and y = 3.2 -> [0.8, 3.2]

        return: the encoding using fourier cos base coding
        """
        assert num_dims == len(features), "Dimensionality mismatch"

        norm_features = (features - n1) / n2

        codings = np.cos(np.matmul(norm_features, coefs) * np.pi)
        return codings

    return discretizer


if __name__ == '__main__':
    fr = [[5.,10],[-10,10]]
    bn = [5,5]
    disc = get_discretizer(fr,bn,True)
    print(disc([10,10]))
