"""
=====================================
Gaussian mixture models(混合高斯模型）
=====================================

不同协方差矩阵参数：
    'full'：每个component独自训练
        each component has its own general covariance matrix
    'tied'：参数共享
        all components share the same general covariance matrix
    'diag'：：训练成和坐标平行
        each component has its own diagonal covariance matrix
    'spherical'：训练成球形
        each component has its own single variance

模型选择：
    利用bic评分，确定component个数和协方差矩阵参数

变体+正则化（BayesianGaussianMixture - 无限混合模型）：
    增加狄利克雷过程先验知识
    可以减少不必要的component

"""


def plot_gmm_covariances():
    """
    ===============
    GMM covariances
    ===============

    Demonstration of several covariances types for Gaussian mixture models.

    See :ref:`gmm` for more information on the estimator.

    Although GMM are often used for clustering, we can compare the obtained
    clusters with the actual classes from the dataset. We initialize the means
    of the Gaussians with the means of the classes from the training set to make
    this comparison valid.

    We plot predicted labels on both training and held out test data using a
    variety of GMM covariance types on the iris dataset.
    We compare GMMs with spherical, diagonal, full, and tied covariance
    matrices in increasing order of performance. Although one would
    expect full covariance to perform best in general, it is prone to
    overfitting on small datasets and does not generalize well to held out
    test data.

    On the plots, train data is shown as dots, while test data is shown as
    crosses. The iris dataset is four-dimensional. Only the first two
    dimensions are shown here, and thus some points are separated in other
    dimensions.
    """

    # Author: Ron Weiss <ronweiss@gmail.com>, Gael Varoquaux
    # Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
    # License: BSD 3 clause

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    import numpy as np

    from sklearn import datasets
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import StratifiedKFold

    print(__doc__)

    colors = ['navy', 'turquoise', 'darkorange']

    def make_ellipses(gmm, ax):
        for _n, _color in enumerate(colors):
            if gmm.covariance_type == 'full':
                covariances = gmm.covariances_[_n][:2, :2]
            elif gmm.covariance_type == 'tied':
                covariances = gmm.covariances_[:2, :2]
            elif gmm.covariance_type == 'diag':
                covariances = np.diag(gmm.covariances_[_n][:2])
            elif gmm.covariance_type == 'spherical':
                covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[_n]
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            ell = mpl.patches.Ellipse(gmm.means_[_n, :2], v[0], v[1],
                                      180 + angle, color=_color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

    iris = datasets.load_iris()

    # Break up the dataset into non-overlapping training (75%) and testing
    # (25%) sets.
    skf = StratifiedKFold(n_splits=4)
    # Only take the first fold.
    train_index, test_index = next(iter(skf.split(iris.data, iris.target)))

    X_train = iris.data[train_index]
    y_train = iris.target[train_index]
    X_test = iris.data[test_index]
    y_test = iris.target[test_index]

    n_classes = len(np.unique(y_train))

    # Try GMMs using different types of covariances.
    estimators = dict((cov_type, GaussianMixture(n_components=n_classes,
                                                 covariance_type=cov_type, max_iter=20, random_state=0))
                      for cov_type in ['spherical', 'diag', 'tied', 'full'])

    n_estimators = len(estimators)

    plt.figure(figsize=(3 * n_estimators // 2, 6))
    plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                        left=.01, right=.99)

    for index, (name, estimator) in enumerate(estimators.items()):
        # Since we have class labels for the training data, we can
        # initialize the GMM parameters in a supervised manner.
        estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                         for i in range(n_classes)])

        # Train the other parameters using the EM algorithm.
        estimator.fit(X_train)

        h = plt.subplot(2, n_estimators // 2, index + 1)
        make_ellipses(estimator, h)

        for n, color in enumerate(colors):
            data = iris.data[iris.target == n]
            plt.scatter(data[:, 0], data[:, 1], s=0.8, color=color,
                        label=iris.target_names[n])
        # Plot the test data with crosses
        for n, color in enumerate(colors):
            data = X_test[y_test == n]
            plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)

        y_train_pred = estimator.predict(X_train)
        train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
        plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
                 transform=h.transAxes)

        y_test_pred = estimator.predict(X_test)
        test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
        plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
                 transform=h.transAxes)

        plt.xticks(())
        plt.yticks(())
        plt.title(name)

    plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))

    plt.show()


def plot_gmm_pdf():
    """
    =========================================
    Density Estimation for a Gaussian mixture
    =========================================

    Plot the density estimation of a mixture of two Gaussians. Data is
    generated from two Gaussians with different centers and covariance
    matrices.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from sklearn import mixture

    n_samples = 300

    # generate random sample, two components
    np.random.seed(0)

    # generate spherical data centered on (20, 20)
    shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

    # generate zero centered stretched Gaussian data
    C = np.array([[0., -0.7], [3.5, .7]])
    stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

    # concatenate the two datasets into the final training set
    X_train = np.vstack([shifted_gaussian, stretched_gaussian])

    # fit a Gaussian Mixture Model with two components
    clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
    clf.fit(X_train)

    # display predicted scores by the model as a contour plot
    x = np.linspace(-20., 30.)
    y = np.linspace(-20., 40.)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    Z = Z.reshape(X.shape)

    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                     levels=np.logspace(0, 3, 10))
    plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(X_train[:, 0], X_train[:, 1], .8)

    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    plt.show()


def plot_gmm_selection():
    """
    ================================
    Gaussian Mixture Model Selection
    ================================

    This example shows that model selection can be performed with
    Gaussian Mixture Models using information-theoretic criteria (BIC).
    Model selection concerns both the covariance type
    and the number of components in the model.
    In that case, AIC also provides the right result (not shown to save time),
    but BIC is better suited if the problem is to identify the right model.
    Unlike Bayesian procedures, such inferences are prior-free.

    In that case, the model with 2 components and full covariance
    (which corresponds to the true generative model) is selected.
    """

    import numpy as np
    import itertools

    from scipy import linalg
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    from sklearn import mixture

    print(__doc__)

    # Number of samples per component
    n_samples = 500

    # Generate random sample, two components
    np.random.seed(0)
    C = np.array([[0., -0.1], [1.7, .4]])
    X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
              .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 7)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
           .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)

    # Plot the winner
    splot = plt.subplot(2, 1, 2)
    Y_ = clf.predict(X)
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                               color_iter)):
        v, w = linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title('Selected GMM: full model, 2 components')
    plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.show()


def plot_gmm():
    """
    =================================
    Gaussian Mixture Model Ellipsoids
    =================================
    对比贝叶斯高斯混合模型，增加狄利克雷过程先验知识

    Plot the confidence ellipsoids of a mixture of two Gaussians
    obtained with Expectation Maximisation (``GaussianMixture`` class) and
    Variational Inference (``BayesianGaussianMixture`` class models with
    a Dirichlet process prior).

    Both models have access to five components with which to fit the data. Note
    that the Expectation Maximisation model will necessarily use all five
    components while the Variational Inference model will effectively only use as
    many as are needed for a good fit. Here we can see that the Expectation
    Maximisation model splits some components arbitrarily, because it is trying to
    fit too many components, while the Dirichlet Process model adapts it number of
    state automatically.

    This example doesn't show it, as we're in a low-dimensional space, but
    another advantage of the Dirichlet process model is that it can fit
    full covariance matrices effectively even when there are less examples
    per cluster than there are dimensions in the data, due to
    regularization properties of the inference algorithm.
    """

    import itertools

    import numpy as np
    from scipy import linalg
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    from sklearn import mixture

    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                                  'darkorange'])

    def plot_results(X, Y_, means, covariances, index, title):
        splot = plt.subplot(2, 1, 1 + index)
        for i, (mean, covar, color) in enumerate(zip(
                means, covariances, color_iter)):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)

        plt.xlim(-9., 5.)
        plt.ylim(-3., 6.)
        plt.xticks(())
        plt.yticks(())
        plt.title(title)

    # Number of samples per component
    n_samples = 500

    # Generate random sample, two components
    np.random.seed(0)
    C = np.array([[0., -0.1], [1.7, .4]])
    X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
              .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

    # Fit a Gaussian mixture with EM using five components
    gmm = mixture.GaussianMixture(n_components=5, covariance_type='full').fit(X)
    plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
                 'Gaussian Mixture')

    # Fit a Dirichlet process Gaussian mixture using five components
    dpgmm = mixture.BayesianGaussianMixture(n_components=5,
                                            covariance_type='full').fit(X)
    plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
                 'Bayesian Gaussian Mixture with a Dirichlet process prior')

    plt.show()


if __name__ == '__main__':
    plot_gmm_pdf()
