"""
支持向量机：
    优点：
        在高维空间有效。
        在维度数量大于样本数量的情况下仍然有效。
        在决策功能（称为支持向量）中使用训练点的子集，因此它也是内存有效的。
        多功能：可以为决策功能指定不同的内核函数。提供通用内核，但也可以指定自定义内核
    缺点：
        如果特征数量远远大于样本数量，则该方法可能会导致较差的性能。
        支持向量机不直接提供概率估计


    rbf函数中的参数（gama，C):
        gama:
            参数gamma定义了单个训练样本的影响大小，值越小影响越大，值越大影响越小。
            参数gamma可以看作被模型选中作为支持向量的样本的影响半径的倒数
            Intuitively, the ``gamma`` parameter defines how far the influence of a single
            training example reaches, with low values meaning 'far' and high values meaning
            'close'. The ``gamma`` parameters can be seen as the inverse of the radius of
            influence of samples selected by the model as support vectors.

        C：
            正则化系数，参数C在误分类样本和分界面简单性之间进行权衡。
            低的C值使分界面平滑，而高的C值通过增加模型自由度以选择更多支持向量来确保所有样本都被正确分类
            C无穷大，意味着没有软间隔
            `C`` behaves as a regularization parameter in the SVM

    SVC,NuSVC,LinearSVC:
        1、多分类训练方式不同1 vs 1,1 vs rest：
            SVC: 1 vs 1
        2、损失函数不同：
            ``LinearSVC`` minimizes the squared hinge loss while
            ``SVC`` minimizes the regular hinge loss.
        3、NuSVC带参数的svc:
            可以选择支持向量数量
            Similar to SVC but uses a parameter to control the number of support vectors.

    复杂度：
        n_features * n_samples**2
        n_features * n_samples**3
"""
from sklearn import svm


def svc_prdct():
    """
    svc 预测,二分类
    :return:
    """
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC(gamma='scale')
    clf.fit(X, y)
    print(clf.predict([[2., 2.]]))
    print(clf.support_vectors_)
    print(clf.support_)
    print(clf.n_support_)


def svc_prdct_multi():
    """
    svc 预测多分类
    如果n_class是类的数量，那么构造 n_class * (n_class - 1) / 2 个分类器
    :return:
    """
    X = [[0], [1], [2], [3]]
    Y = [0, 1, 2, 3]
    clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
    clf.fit(X, Y)

    dec = clf.decision_function([[1]])
    print(dec.shape[1])  # 4 classes: 4*3/2 = 6

    clf.decision_function_shape = "ovr"
    dec = clf.decision_function([[1]])
    print(dec.shape[1])  # 4 classes


def plot_separating_hyperplane():
    """
    SVM的超平面
    :return:
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm
    from sklearn.datasets import make_blobs

    # we create 40 separable points
    X, y = make_blobs(n_samples=40, centers=2, random_state=6)

    # fit the model, don't regularize for illustration purposes
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)

    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.show()


def plot_iris():
    """
    ==================================================
    Plot different SVM classifiers in the iris dataset
    ==================================================
    LinearSVC与SVC(kernel='linear')

    Comparison of different linear SVM classifiers on a 2D projection of the iris
    dataset. We only consider the first 2 features of this dataset:

    - Sepal length
    - Sepal width

    This example shows how to plot the decision surface for four SVM classifiers
    with different kernels.

    The linear models ``LinearSVC()`` and ``SVC(kernel='linear')`` yield slightly
    different decision boundaries. This can be a consequence of the following
    differences:
    LinearSVC与SVC(kernel='linear')的区别在于损失函数不同，多分类处理方法不同

    - ``LinearSVC`` minimizes the squared hinge loss while ``SVC`` minimizes the
      regular hinge loss.

    - ``LinearSVC`` uses the One-vs-All (also known as One-vs-Rest) multiclass
      reduction while ``SVC`` uses the One-vs-One multiclass reduction.

    Both linear models have linear decision boundaries (intersecting hyperplanes)
    while the non-linear kernel models (polynomial or Gaussian RBF) have more
    flexible non-linear decision boundaries with shapes that depend on the kind of
    kernel and its parameters.

    .. NOTE:: while plotting the decision function of classifiers for toy 2D
       datasets can help get an intuitive understanding of their respective
       expressive power, be aware that those intuitions don't always generalize to
       more realistic high-dimensional problems.

    """
    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm, datasets

    def make_meshgrid(x, _y, h=.02):
        """Create a mesh of points to plot in

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        _y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = _y.min() - 1, _y.max() + 1
        _xx, _yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
        return _xx, _yy

    def plot_contours(_ax, clf, _xx, _yy, **params):
        """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        _ax: matplotlib axes object
        clf: a classifier
        _xx: meshgrid ndarray
        _yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
        """
        Z = clf.predict(np.c_[_xx.ravel(), _yy.ravel()])
        Z = Z.reshape(_xx.shape)
        out = _ax.contourf(_xx, _yy, Z, **params)
        return out

    # import some data to play with
    iris = datasets.load_iris()
    # Take the first two features. We could avoid this by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 1.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C),
              svm.SVC(kernel='rbf', gamma=0.7, C=C),
              svm.SVC(kernel='poly', degree=3, C=C))
    models = (clf.fit(X, y) for clf in models)

    # title for the plots
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()


def plot_rbf_parameters():
    """
    ==================
    RBF SVM parameters
    ==================

    This example illustrates the effect of the parameters ``gamma`` and ``C`` of
    the Radial Basis Function (RBF) kernel SVM.

    Intuitively, the ``gamma`` parameter defines how far the influence of a single
    training example reaches, with low values meaning 'far' and high values meaning
    'close'. The ``gamma`` parameters can be seen as the inverse of the radius of
    influence of samples selected by the model as support vectors.

    The ``C`` parameter trades off correct classification of training examples
    against maximization of the decision function's margin. For larger values of
    ``C``, a smaller margin will be accepted if the decision function is better at
    classifying all training points correctly. A lower ``C`` will encourage a
    larger margin, therefore a simpler decision function, at the cost of training
    accuracy. In other words``C`` behaves as a regularization parameter in the
    SVM.

    The first plot is a visualization of the decision function for a variety of
    parameter values on a simplified classification problem involving only 2 input
    features and 2 possible target classes (binary classification). Note that this
    kind of plot is not possible to do for problems with more features or target
    classes.

    The second plot is a heatmap of the classifier's cross-validation accuracy as a
    function of ``C`` and ``gamma``. For this example we explore a relatively large
    grid for illustration purposes. In practice, a logarithmic grid from
    :math:`10^{-3}` to :math:`10^3` is usually sufficient. If the best parameters
    lie on the boundaries of the grid, it can be extended in that direction in a
    subsequent search.

    Note that the heat map plot has a special colorbar with a midpoint value close
    to the score values of the best performing models so as to make it easy to tell
    them apart in the blink of an eye.

    The behavior of the model is very sensitive to the ``gamma`` parameter. If
    ``gamma`` is too large, the radius of the area of influence of the support
    vectors only includes the support vector itself and no amount of
    regularization with ``C`` will be able to prevent overfitting.

    When ``gamma`` is very small, the model is too constrained and cannot capture
    the complexity or "shape" of the data. The region of influence of any selected
    support vector would include the whole training set. The resulting model will
    behave similarly to a linear model with a set of hyperplanes that separate the
    centers of high density of any pair of two classes.

    For intermediate values, we can see on the second plot that good models can
    be found on a diagonal of ``C`` and ``gamma``. Smooth models (lower ``gamma``
    values) can be made more complex by increasing the importance of classifying
    each point correctly (larger ``C`` values) hence the diagonal of good
    performing models.

    Finally one can also observe that for some intermediate values of ``gamma`` we
    get equally performing models when ``C`` becomes very large: it is not
    necessary to regularize by enforcing a larger margin. The radius of the RBF
    kernel alone acts as a good structural regularizer. In practice though it
    might still be interesting to simplify the decision function with a lower
    value of ``C`` so as to favor models that use less memory and that are faster
    to predict.

    We should also note that small differences in scores results from the random
    splits of the cross-validation procedure. Those spurious variations can be
    smoothed out by increasing the number of CV iterations ``n_splits`` at the
    expense of compute time. Increasing the value number of ``C_range`` and
    ``gamma_range`` steps will increase the resolution of the hyper-parameter heat
    map.

    """
    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_iris
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.model_selection import GridSearchCV

    # Utility function to move the midpoint of a colormap to be around
    # the values of interest.

    class MidpointNormalize(Normalize):

        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, _y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, _y))

    # #############################################################################
    # Load and prepare data set
    #
    # dataset for grid search

    iris = load_iris()
    X = iris.data
    y = iris.target

    # Dataset for decision function visualization: we only keep the first two
    # features in X and sub-sample the dataset to keep only 2 classes and
    # make it a binary classification problem.

    X_2d = X[:, :2]
    X_2d = X_2d[y > 0]
    y_2d = y[y > 0]
    y_2d -= 1

    # It is usually a good idea to scale the data for SVM training.
    # We are cheating a bit in this example in scaling all of the data,
    # instead of fitting the transformation on the training set and
    # just applying it on the test set.

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_2d = scaler.fit_transform(X_2d)

    # #############################################################################
    # Train classifiers
    #
    # For an initial search, a logarithmic grid with basis
    # 10 is often helpful. Using a basis of 2, a finer
    # tuning can be achieved but at a much higher cost.

    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

    # Now we need to fit a classifier for all parameters in the 2d version
    # (we use a smaller set of parameters here because it takes a while to train)

    C_2d_range = [1e-2, 1, 1e2]
    gamma_2d_range = [1e-1, 1, 1e1]
    classifiers = []
    for C in C_2d_range:
        for gamma in gamma_2d_range:
            clf = SVC(C=C, gamma=gamma)
            clf.fit(X_2d, y_2d)
            classifiers.append((C, gamma, clf))

    # #############################################################################
    # Visualization
    #
    # draw visualization of parameter effects

    plt.figure(figsize=(8, 6))
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    for (k, (C, gamma, clf)) in enumerate(classifiers):
        # evaluate decision function in a grid
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # visualize decision function for these parameters
        plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
        plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
                  size='medium')

        # visualize parameter's effect on decision function
        plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
                    edgecolors='k')
        plt.xticks(())
        plt.yticks(())
        plt.axis('tight')

    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                         len(gamma_range))

    # Draw heatmap of the validation accuracy as a function of gamma and C
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()


def separating_hyperplane_for_unbalanced_classes():
    """
    样本不均衡分类, 可以尝试把核函数修改为高斯核'rbf'
    :return:
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm
    from sklearn.datasets import make_blobs

    # we create two clusters of random points
    n_samples_1 = 1000
    n_samples_2 = 100
    centers = [[0.0, 0.0], [2.0, 2.0]]
    clusters_std = [1.5, 0.5]
    X, y = make_blobs(n_samples=[n_samples_1, n_samples_2],
                      centers=centers,
                      cluster_std=clusters_std,
                      random_state=0, shuffle=False)

    # fit the model and get the separating hyperplane
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X, y)

    # fit the model and get the separating hyperplane using weighted classes
    wclf = svm.SVC(kernel='linear', class_weight={1: 10})
    wclf.fit(X, y)

    # plot the samples
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

    # plot the decision functions for both classifiers
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # get the separating hyperplane
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    a = ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

    # get the separating hyperplane for weighted classes
    Z = wclf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins for weighted classes
    b = ax.contour(XX, YY, Z, colors='r', levels=[0], alpha=0.5, linestyles=['-'])

    plt.legend([a.collections[0], b.collections[0]], ["non weighted", "weighted"],
               loc="upper right")
    plt.show()


def plot_svm_nonlinear():
    """
    ======================
    Non-linear SVM 异或问题
    ======================

    Perform binary classification using non-linear SVC
    with RBF kernel. The target to predict is a XOR of the
    inputs.

    The color map illustrates the decision function learned by the SVC.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm

    xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                         np.linspace(-3, 3, 500))
    np.random.seed(0)
    X = np.random.randn(300, 2)
    Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

    # fit the model
    clf = svm.NuSVC()
    clf.fit(X, Y)

    # plot the decision function for each datapoint on the grid
    # np._c 按列排列[列1，列2],相当于vstack([列1,列2]).T
    # Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = clf.decision_function(np.vstack([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)

    # 绘制热力图
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
               origin='lower', cmap=plt.cm.PuOr_r)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, linetypes='--')
    plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis([-3, 3, -3, 3])
    plt.show()


def plot_weighted_samples():
    """
    =====================
    SVM: Weighted samples
    =====================

    Plot decision function of a weighted dataset, where the size of points
    is proportional to its weight.

    The sample weighting rescales the C parameter, which means that the classifier
    puts more emphasis on getting these points right. The effect might often be
    subtle.
    To emphasize the effect here, we particularly weight outliers, making the
    deformation of the decision boundary very visible.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm

    def plot_decision_function(classifier, sample_weight, axis, title):
        # plot the decision function
        xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))

        Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # plot the line, the points, and the nearest vectors to the plane
        axis.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
        axis.scatter(X[:, 0], X[:, 1], c=y, s=100 * sample_weight, alpha=0.9,
                     cmap=plt.cm.bone, edgecolors='black')

        axis.axis('off')
        axis.set_title(title)

    # we create 20 points
    np.random.seed(0)
    X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]
    print(np.vstack([np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]))
    print(X)
    y = [1] * 10 + [-1] * 10
    sample_weight_last_ten = abs(np.random.randn(len(X)))
    sample_weight_constant = np.ones(len(X))
    # and bigger weights to some outliers
    sample_weight_last_ten[15:] *= 5
    sample_weight_last_ten[9] *= 15

    # for reference, first fit without sample weights

    # fit the model
    clf_weights = svm.SVC(gamma=1)
    clf_weights.fit(X, y, sample_weight=sample_weight_last_ten)

    clf_no_weights = svm.SVC(gamma=1)
    clf_no_weights.fit(X, y)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_decision_function(clf_no_weights, sample_weight_constant, axes[0],
                           "Constant weights")
    plot_decision_function(clf_weights, sample_weight_last_ten, axes[1],
                           "Modified weights")

    plt.show()


def plot_svm_regression():
    """
    支持向量机回归模型
    ===================================================================
    Support Vector Regression (SVR) using linear and non-linear kernels
    ===================================================================

    Toy example of 1D regression using linear, polynomial and RBF kernels.

    """
    print(__doc__)

    import numpy as np
    from sklearn.svm import SVR
    import matplotlib.pyplot as plt

    # #############################################################################
    # Generate sample data
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    y = np.sin(X).ravel()

    # #############################################################################
    # Add noise to targets
    y[::5] += 3 * (0.5 - np.random.rand(8))

    # #############################################################################
    # Fit regression model
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(X, y).predict(X)
    y_lin = svr_lin.fit(X, y).predict(X)
    y_poly = svr_poly.fit(X, y).predict(X)

    # #############################################################################
    # Look at the results
    lw = 2
    plt.scatter(X, y, color='darkorange', label='data')
    plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
    plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
    plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()


def plot_custom_kernel():
    """
    ======================
    SVM with custom kernel
    ======================
    自定义核函数
    Simple usage of Support Vector Machines to classify a sample. It will
    plot the decision surface and the support vectors.

    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm, datasets

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features. We could
    # avoid this ugly slicing by using a two-dim dataset
    Y = iris.target

    def my_kernel(_X, _Y):
        """
        We create a custom kernel:

                     (2  0)
        k(X, Y) = X  (    ) Y.T
                     (0  1)
        """
        M = np.array([[2, 0], [0, 1.0]])
        return np.dot(np.dot(_X, M), _Y.T)

    h = .02  # step size in the mesh

    # we create an instance of SVM and fit out data.
    clf = svm.SVC(kernel=my_kernel)
    clf.fit(X, Y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
    plt.title('3-Class classification using Support Vector Machine with custom'
              ' kernel')
    plt.axis('tight')
    plt.show()


if __name__ == '__main__':
    # svc_prdct()
    plot_separating_hyperplane()
    # svc_prdct_multi()
    # plot_iris()
    # plot_rbf_parameters()
    separating_hyperplane_for_unbalanced_classes()
    # plot_svm_nonlinear()
    # plot_weighted_samples()
    plot_svm_regression()
    # plot_custom_kernel()
