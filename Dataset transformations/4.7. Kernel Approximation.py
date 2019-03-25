"""
==================================
4.7. Kernel Approximation（核方法）
=================================
rbf核：
    Approximates feature map of an RBF kernel by Monte Carlo approximation
        of its Fourier transform.
    kernel_approximation.RBFSampler(gamma=1, random_state=1).fit_transform

"""


def example():
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import SGDClassifier
    X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    y = [0, 0, 1, 1]
    rbf_feature = RBFSampler(gamma=1, random_state=1)
    X_features = rbf_feature.fit_transform(X)
    clf = SGDClassifier(max_iter=5)
    print(clf.fit(X_features, y))

    print(clf.score(X_features, y))
