"""
polynomial based coding

MM, 2023
"""

import numpy as np
from itertools import product
from functools import reduce
from operator import mul


def get_discretizer(orders):
    """
    feature_ranges: range of each feature
        example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
    number_orders: order size for each dimension
        example: 8 orders for x and 6 orders for y -> [8, 6]

    return: rbf coder
    """

    num_dims = len(orders)

    power_ranges = [list(range(x)) for x in orders]

    def discretizer(features, **kwargs):
        """
        feature: sample with multiple dimensions to be encoded;
            example: x = 0.8 and y = 3.2 -> [0.8, 3.2]

        return: the encoding using rbf coding
        """
        assert num_dims == len(features), "Dimensionality mismatch"

        x = [ reduce(mul, [features[i]**p for i, p in enumerate(powers)], 1)
            for powers in product(*power_ranges)
        ]

        return np.array(x)

    return discretizer

