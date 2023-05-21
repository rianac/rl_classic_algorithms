"""
Bins' based coding

MM, 2023
"""

import numpy as np
from functools import reduce
from operator import mul



def get_discretizer(feature_ranges, number_bins):
    """
    feature_ranges: range of each feature
        example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
    number_bins: bin size for each dimension
        example: 8 bins for x and 6 bins for y -> [8, 6]

    return: bin coder
    """

    num_dims = len(feature_ranges)
    assert num_dims == len(number_bins), "Dimensionality mismatch"

    # Find separators for separating bins along all dimensions.
    bin_separators = [
        np.linspace(feat_range[0], feat_range[1], feat_bins + 1)[1:-1]
        for feat_range, feat_bins in zip(feature_ranges, number_bins)
    ]

    def discretizer(features, vector_type=False):
        """
        feature: sample with multiple dimensions to be encoded;
            example: x = 0.8 and y = 3.2 -> [0.8, 3.2]

        return: the encoding for the feature on each dimension
        """
        assert num_dims == len(features), "Dimensionality mismatch"

        # Select suitable bins for feature sample (for each dimenstion
        # separately).
        feat_codings = tuple(map(np.digitize, features, bin_separators))

        # Transform indices of selected bins for separate dimensions into
        # one vector if required.
        if vector_type:
            x = np.zeros(number_bins)
            x[feat_codings] = 1
            feat_codings = x.reshape(-1)
        else:
            feat_codings = [feat_codings]

        return feat_codings

    return discretizer

