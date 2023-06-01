"""
RBF based coding

MM, 2023
"""

import numpy as np
from itertools import product



def get_discretizer(feature_ranges, number_centers):
    """
    feature_ranges: range of each feature
        example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
    number_centers: bin size for each dimension
        example: 8 bins for x and 6 bins for y -> [8, 6]

    return: rbf coder
    """

    num_dims = len(feature_ranges)
    assert num_dims == len(number_centers), "Dimensionality mismatch"

    # Find centers of all bins along all dimensions.
    bin_centers = [
        np.linspace(feat_range[0], feat_range[1], 2*feat_centers + 1)[1:-1:2]
        for feat_range, feat_centers in zip(feature_ranges, number_centers)
    ]

    # Find widths of bins along all dimensions
    bin_widths = (
        (feat_range[1] - feat_range[0])/feat_centers
        for feat_range, feat_centers in zip(feature_ranges, number_centers)
    )

    # Prepare denominators for rbf 
    denominators = [2 * sigma**2 for sigma in bin_widths]

    def discretizer(features, **kwargs):
        """
        feature: sample with multiple dimensions to be encoded;
            example: x = 0.8 and y = 3.2 -> [0.8, 3.2]

        return: the encoding using rbf coding
        """
        assert num_dims == len(features), "Dimensionality mismatch"

        def gauss(features, center):
            x = (
                (f - c)**2 / d
                for f, c, d in zip(features, center, denominators)
            )
            return np.exp(- sum(x))

        x =  [gauss(features, center) for center in product(*bin_centers)]

        return np.array(x)

    return discretizer


if __name__ == '__main__':
    fr = [[0.,10],[0,10]]
    bn = [5,5]
    disc = get_discretizer(fr,bn)
    print(disc([1,1]))
