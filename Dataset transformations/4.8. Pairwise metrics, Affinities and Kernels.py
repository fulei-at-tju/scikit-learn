"""
==============================================
4.8. Pairwise metrics, Affinities and Kernels
==============================================
 sklearn.metrics.pairwise_distances 代表两两距离
 sklearn.metrics.pairwise_kernels 代表两两相似度

 sklearn.metrics.pairwise
"""


def example():
    import numpy as np
    from sklearn.metrics import pairwise_distances
    from sklearn.metrics import pairwise_kernels
    X = np.array([[2, 3], [3, 5], [5, 8]])
    Y = np.array([[1, 0], [2, 1]])

    print(pairwise_distances(X, Y, metric='manhattan'))
    print(pairwise_distances(X, metric='manhattan'))
    print(pairwise_kernels(X, Y, metric='linear'))


if __name__ == '__main__':
    pass

