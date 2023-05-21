"""
Tiling software based on Sutton-Barto, 2018.

MM, 2023
"""

import numpy as np
from functools import reduce
from operator import mul, add



def create_tilings(num_dims, feature_ranges, number_tilings, number_bins):
    """
    num_dims: dimensionality of state space
        example: two dimensional state space -> 2
    feature_ranges: range of each feature
        example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
    number_tilings: number of tilings
        example: 8 tilings -> 8
    number_bins: bin size for each dimension
        example: 8 bins for x and 6 bins for y -> [8, 6]

    return: separators for separating bins for all tilings along all
            dimensions
    """
    def transform(op, tiling, coefs):
        """
        Helper for pairwise tiling transformation (translation or
        dilatation).
           example: add, [[3,4],[5,6]], [1,2] -> [[4,5],[7,8]]
           example: mul, [[3,4],[5,6]], [2,3] -> [[6,8],[15,18]]
        """
        return [[op(x,coef) for x in feat_tiling]
                for feat_tiling, coef in zip(tiling, coefs)]

    # Each side of state space divided into slices - all displacements
    # will be later defined in number of slices.
    # Slice lengths vary based on ranges of different dimensions.
    feat_slice_lengths = [
        (feat_range[1] - feat_range[0]) /
        ((feat_bins - 1) * number_tilings + 1)
        for feat_range, feat_bins in zip(feature_ranges, number_bins)
    ]

    # Separators for separating bins along all dimensions, initially
    # each bin has size equal number of tillings.
    tiling_template = [
        np.linspace(0, number_tilings * feat_bins, feat_bins + 1)[1:-1]
        for feat_bins in number_bins
    ]

    # Shifts for displacement all tilings in relation to one another.
    # As shift increments first odd integers are used.
    shift_pattern = [2 * i + 1 for i in range(num_dims)]
    shifts = (((til * shift_pattern[i]) % number_tilings
               for i in range(num_dims)) for til in range(number_tilings))

    # All tilings generated based on template for separating bins
    # and approprately translated in relations to one another. Bin 
    # sizes not respecting feature ranges along dimensions.
    tilings = (transform(add, tiling_template, shift) for shift in shifts)

    # Bin sizes modified for them to respect slice lengths along
    # different dimensions (bins dilatation).
    tilings = (transform(mul, tiling, feat_slice_lengths) for tiling in tilings)

    # Bin positions modified for them to respect feature ranges along
    # different dimensions - aligning with minimal range values (bins
    # translated).
    tilings = (transform(add, tiling, [fr[0] for fr in feature_ranges])
               for tiling in tilings)

    # Final positioning (translation) of bins (based on sutton-barto
    # 2ed, fig 9.9)
    [shifts] = transform(mul, [feat_slice_lengths],
                             [-(number_tilings - 1)] * num_dims)
    tilings = [transform(add, tiling, shifts) for tiling in tilings]

    return tilings


def get_discretizer(feature_ranges, number_tilings, number_bins):
    """
    feature_ranges: range of each feature
        example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
    number_tilings: number of tilings
        example: 8 tilings -> 8
    number_bins: bin size for each dimension
        example: 8 bins for x and 6 bins for y -> [8, 6]

    return: tile coder
    """

    # Recommended number of tilings (sutton-barto 2ed, p. 220)
    # - in relation with number of dimensions
    # - beeing power of 2
    num_dims = len(feature_ranges)
    assert number_tilings >= 4 * num_dims, "Number of tilings too small"
    assert (number_tilings & (number_tilings - 1) == 0
            and number_tilings != 0), "Number of tilings not power of 2"

    assert num_dims == len(number_bins), "Dimensionality mismatch"

    # Find separators for separating bins for all tilings along all
    # dimensions.
    tilings = create_tilings(num_dims, feature_ranges, number_tilings,
                             number_bins)

    # Weights for converting multidimensional position of bins into
    # one indice.
    bin_weights = number_bins[1:] + [1]
    bin_weights = [reduce(mul, bin_weights[i:], 1) for i in range(num_dims)]

    def discretizer(features, vector_type=False):
        """
        feature: sample with multiple dimensions to be encoded;
            example: x = 0.8 and y = 3.2 -> [0.8, 3.2]
        output_type: the form of a list of indices or a vector of values

        return: the encoding for the feature on each tiling
        """
        assert num_dims == len(features), "Dimensionality mismatch"

        # Select suitable bins for feature sample (for each dimension
        # separately).
        feat_codings = list(map(lambda x: tuple(map(np.digitize, features, x)),
                                tilings))

        # Transform indices of selected bins for separate dimensions into
        # one vector if required.
        if vector_type:
            x = [np.zeros(number_bins) for _ in range(number_tilings)]
            for i in range(number_tilings):
                x[i][feat_codings[i]] = 1
            feat_codings = np.concatenate([y.reshape(-1) for y in x])

        return feat_codings

    return discretizer


if __name__ == '__main__':
    import unittest
                
    class TestCreateTilings(unittest.TestCase):
        def test_dimensions(self):
            ranges = [[0,1],[1,2],[2,3]]
            self.assertTrue(
                np.allclose(create_tilings(1,ranges[1:2],1,[2]),
                            [[[1.5]]])
            )
            self.assertTrue(
                np.allclose(create_tilings(2,ranges[:2],1,[2,2]),
                            [[[0.5],[1.5]]])
            )
            self.assertTrue(
                np.allclose(create_tilings(3,ranges,1,[2,2,2]),
                            [[[0.5],[1.5],[2.5]]])
            )

        def test_1_tiling(self):
            ranges = [[-1,1],[0,1]]
            self.assertTrue(
                np.allclose(create_tilings(2,ranges,1,[2,2]),
                            [[[0.0],[0.5]]])
            )
            self.assertTrue(
                np.allclose(create_tilings(2,ranges,1,[4,4]),
                            [[[-0.5,0.0,0.5],
                              [0.25,0.5,0.75]]])
            )

        def test_n_tilings(self):
            ranges = [[0,1],[0,1]]
            self.assertTrue(
                np.allclose(create_tilings(2,ranges,3,[2,2]),
                            [[[0.25],[0.25]],[[0.5],[0.25]],
                             [[0.75],[0.25]]])
            )
            self.assertTrue(
                np.allclose(create_tilings(2,ranges,4,[2,2]),
                            [[[0.2],[0.2]],[[0.4],[0.8]],
                             [[0.6],[0.6]],[[0.8],[0.4]]])
            )

    class TestGetDiscretizer(unittest.TestCase):
        def test_discretize_1_dim(self):
            disc = get_discretizer([[0,1]],4,[2])
            self.assertEqual(list(disc([0.1])),[(0,),(0,),(0,),(0,)])
            self.assertEqual(list(disc([0.3])),[(1,),(0,),(0,),(0,)])
            self.assertEqual(list(disc([0.5])),[(1,),(1,),(0,),(0,)])
            self.assertEqual(list(disc([0.7])),[(1,),(1,),(1,),(0,)])
            self.assertEqual(list(disc([0.9])),[(1,),(1,),(1,),(1,)])
            
        def test_discretize_2_dim(self):
            ranges = [[0,1],[0,1]]
            disc = get_discretizer(ranges,8,[2,2])
            self.assertEqual(list(disc([0.9,0.1])),
                             [(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0)])
            self.assertEqual(list(disc([0.9,0.9])),
                             [(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1),(1,1)])
            self.assertEqual(list(disc([0.1,0.1])),
                             [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)])
            self.assertEqual(list(disc([0.1,0.9])),
                             [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)])
           
    unittest.main()

