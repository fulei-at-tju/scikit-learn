"""
线性和二次判别分析
LDA（线性判别式）和QDA(二次判别分析)是两个经典分类器

LDA也可用于降维（dimensionality reduction），类内散度（协方差），类见散度（均值差）
PCA降维,协方差矩阵，前n特征，特征向量， 与原数据集相乘，实现特征降维（高维向低维映射）
"""


def plot_lda_qda():
    """
    ====================================================================
    Linear and Quadratic Discriminant Analysis with covariance ellipsoid
    ====================================================================

    This example plots the covariance ellipsoids of each class and
    decision boundary learned by LDA and QDA. The ellipsoids display
    the double standard deviation for each class. With LDA, the
    standard deviation is the same for all the classes, while each
    class has its own standard deviation with QDA.
    """
    print(__doc__)

    from scipy import linalg
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import colors

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    # #############################################################################
    # Colormap
    cmap = colors.LinearSegmentedColormap(
        'red_blue_classes',
        {'red': [(0, 1, 1), (1, 0.7, 0.7)],
         'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
         'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
    plt.cm.register_cmap(cmap=cmap)

    # #############################################################################
    # Generate datasets
    def dataset_fixed_cov():
        """Generate 2 Gaussians samples with the same covariance matrix"""
        n, dim = 300, 2
        np.random.seed(0)
        C = np.array([[0., -0.23], [0.83, .23]])
        _X = np.r_[np.dot(np.random.randn(n, dim), C),
                   np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
        _y = np.hstack((np.zeros(n), np.ones(n)))
        return _X, _y

    def dataset_cov():
        """Generate 2 Gaussians samples with different covariance matrices"""
        n, dim = 300, 2
        np.random.seed(0)
        C = np.array([[0., -1.], [2.5, .7]]) * 2.
        _X = np.r_[np.dot(np.random.randn(n, dim), C),
                   np.dot(np.random.randn(n, dim), C.T) + np.array([1, 4])]
        _y = np.hstack((np.zeros(n), np.ones(n)))
        return _X, _y

    # #############################################################################
    # Plot functions
    def plot_data(_lda, _X, _y, _y_pred, fig_index):
        _splot = plt.subplot(2, 2, fig_index)
        if fig_index == 1:
            plt.title('Linear Discriminant Analysis')
            plt.ylabel('Data with\n fixed covariance')
        elif fig_index == 2:
            plt.title('Quadratic Discriminant Analysis')
        elif fig_index == 3:
            plt.ylabel('Data with\n varying covariances')

        tp = (_y == _y_pred)  # True Positive
        tp0, tp1 = tp[_y == 0], tp[_y == 1]
        X0, X1 = _X[_y == 0], _X[_y == 1]
        X0_tp, X0_fp = X0[tp0], X0[~tp0]
        X1_tp, X1_fp = X1[tp1], X1[~tp1]

        alpha = 0.5

        # class 0: dots
        plt.plot(X0_tp[:, 0], X0_tp[:, 1], 'o', alpha=alpha,
                 color='red', markeredgecolor='k')
        plt.plot(X0_fp[:, 0], X0_fp[:, 1], '*', alpha=alpha,
                 color='#990000', markeredgecolor='k')  # dark red

        # class 1: dots
        plt.plot(X1_tp[:, 0], X1_tp[:, 1], 'o', alpha=alpha,
                 color='blue', markeredgecolor='k')
        plt.plot(X1_fp[:, 0], X1_fp[:, 1], '*', alpha=alpha,
                 color='#000099', markeredgecolor='k')  # dark blue

        # class 0 and 1 : areas
        nx, ny = 200, 100
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                             np.linspace(y_min, y_max, ny))
        Z = _lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 1].reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                       norm=colors.Normalize(0., 1.))
        plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')

        # means
        plt.plot(_lda.means_[0][0], _lda.means_[0][1],
                 'o', color='black', markersize=10, markeredgecolor='k')
        plt.plot(_lda.means_[1][0], _lda.means_[1][1],
                 'o', color='black', markersize=10, markeredgecolor='k')

        return _splot

    def plot_ellipse(_splot, mean, cov, color):
        v, w = linalg.eigh(cov)
        u = w[0] / linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        # filled Gaussian at 2 standard deviation
        ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                                  180 + angle, facecolor=color,
                                  edgecolor='yellow',
                                  linewidth=2, zorder=2)
        ell.set_clip_box(_splot.bbox)
        ell.set_alpha(0.5)
        _splot.add_artist(ell)
        _splot.set_xticks(())
        _splot.set_yticks(())

    def plot_lda_cov(_lda, _splot):
        plot_ellipse(_splot, _lda.means_[0], _lda.covariance_, 'red')
        plot_ellipse(_splot, _lda.means_[1], _lda.covariance_, 'blue')

    def plot_qda_cov(_qda, _splot):
        plot_ellipse(_splot, _qda.means_[0], _qda.covariance_[0], 'red')
        plot_ellipse(_splot, _qda.means_[1], _qda.covariance_[1], 'blue')

    for i, (X, y) in enumerate([dataset_fixed_cov(), dataset_cov()]):
        # Linear Discriminant Analysis
        lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
        y_pred = lda.fit(X, y).predict(X)
        splot = plot_data(lda, X, y, y_pred, fig_index=2 * i + 1)
        plot_lda_cov(lda, splot)
        plt.axis('tight')

        # Quadratic Discriminant Analysis
        qda = QuadraticDiscriminantAnalysis(store_covariance=True)
        y_pred = qda.fit(X, y).predict(X)
        splot = plot_data(qda, X, y, y_pred, fig_index=2 * i + 2)
        plot_qda_cov(qda, splot)
        plt.axis('tight')
    plt.suptitle('Linear Discriminant Analysis vs Quadratic Discriminant'
                 'Analysis')
    plt.show()


def plot_pca_vs_lda():
    """
    =======================================================
    Comparison of LDA and PCA 2D projection of Iris dataset
    =======================================================

    The Iris dataset represents 3 kind of Iris flowers (Setosa, Versicolour
    and Virginica) with 4 attributes: sepal length, sepal width, petal length
    and petal width.

    Principal Component Analysis (PCA) applied to this data identifies the
    combination of attributes (principal components, or directions in the
    feature space) that account for the most variance in the data. Here we
    plot the different samples on the 2 first principal components.

    Linear Discriminant Analysis (LDA) tries to identify attributes that
    account for the most variance *between classes*. In particular,
    LDA, in contrast to PCA, is a supervised method, using known class labels.
    """
    print(__doc__)

    import matplotlib.pyplot as plt

    from sklearn import datasets
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target
    target_names = iris.target_names

    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)

    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of IRIS dataset')

    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of IRIS dataset')

    plt.show()


def plot_lda():
    """
    ====================================================================
    Normal and Shrinkage Linear Discriminant Analysis for classification
    ====================================================================

    Shows how shrinkage improves classification.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import make_blobs
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    n_train = 20  # samples for training
    n_test = 200  # samples for testing
    n_averages = 50  # how often to repeat classification
    n_features_max = 75  # maximum number of features
    step = 4  # step size for the calculation

    def generate_data(n_samples, _n_features):
        """Generate random blob-ish data with noisy features.

        This returns an array of input data with shape `(n_samples, n_features)`
        and an array of `n_samples` target labels.

        Only one feature contains discriminative information, the other features
        contain only noise.
        """
        _X, _y = make_blobs(n_samples=n_samples, n_features=1, centers=[[-2], [2]])

        # add non-discriminative features
        if _n_features > 1:
            _X = np.hstack([_X, np.random.randn(n_samples, _n_features - 1)])
        return _X, _y

    acc_clf1, acc_clf2 = [], []
    n_features_range = range(1, n_features_max + 1, step)
    for n_features in n_features_range:
        score_clf1, score_clf2 = 0, 0
        for _ in range(n_averages):
            X, y = generate_data(n_train, n_features)

            clf1 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, y)
            clf2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(X, y)

            X, y = generate_data(n_test, n_features)
            score_clf1 += clf1.score(X, y)
            score_clf2 += clf2.score(X, y)

        acc_clf1.append(score_clf1 / n_averages)
        acc_clf2.append(score_clf2 / n_averages)

    features_samples_ratio = np.array(n_features_range) / n_train

    plt.plot(features_samples_ratio, acc_clf1, linewidth=2,
             label="Linear Discriminant Analysis with shrinkage", color='navy')
    plt.plot(features_samples_ratio, acc_clf2, linewidth=2,
             label="Linear Discriminant Analysis", color='gold')

    plt.xlabel('n_features / n_samples')
    plt.ylabel('Supervised learning accuracy')

    plt.legend(loc=1, prop={'size': 12})
    plt.suptitle('Linear Discriminant Analysis vs. \
    shrinkage Linear Discriminant Analysis (1 discriminative feature)')
    plt.show()


if __name__ == '__main__':
    # plot_lda_qda()
    plot_pca_vs_lda()
