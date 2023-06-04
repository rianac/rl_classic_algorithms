import numpy as np
from itertools import product



def get_discretizer(feature_ranges, number_centers, simple=False):
    """
    feature_ranges: range of each feature
        example: x: [-1, 1], y: [2, 5] -> [[-1, 1], [2, 5]]
    number_centers: number of bins for each dimension
        example: 8 bins for x and 6 bins for y -> [8, 6]
    simple: if True then multi-dimensional coding, combination of
        one-dimensional codings otherwise

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
        feature: multi-dimensional somple to be encoded;
            example: x = 0.8 and y = 3.2 -> [0.8, 3.2]

        return: the multi-dimensional encoding using rbf coding
        """
        assert num_dims == len(features), "Dimensionality mismatch"

        # Readable but slow alternative - replaced by a faster one
        # def gauss(features, center):
        #     x = (
        #         (f - c)**2 / d
        #         for f, c, d in zip(features, center, denominators)
        #     )
        #     return np.exp(- sum(x))
        #
        # x =  [gauss(features, center) for center in product(*bin_centers)]
        # x = np.array(x)

        y = np.array([center for center in product(*bin_centers)]).transpose()
        y = [ (features[i] - y[i])**2 / denominators[i]
              for i in range(y.shape[0])]
        y = np.exp(- sum(y))

        return y


    def discretizer_simple(features, **kwargs):
        """
        feature: multi-dimensional sample to be encoded;
            example: x = 0.8 and y = 3.2 -> [0.8, 3.2]

        return: the combination of one-dimensional encodings using rbf coding
        """
        assert num_dims == len(features), "Dimensionality mismatch"

        x = [ np.exp(-(f - c)**2 / d)
              for f, c, d in zip(features, bin_centers, denominators) ]
        x = np.concatenate(x, axis=0)

        return x

    if simple:
        return discretizer_simple
    else:
        return discretizer


if __name__ == '__main__':
    fr = [[0.,10],[0,10]]
    bn = [5,5]
    disc = get_discretizer(fr,bn,True)
    print(disc([1,1]))
    disc = get_discretizer(fr,bn)
    print(disc([1,1]))
