"""
==========================================
Stochastic Gradient Descent （随机梯度下降）
==========================================
可以分类也可以回归：
    SGDClassifier， SGDRegressor

随机梯度下降的优点是：
    高效
    易于实施（很多机会进行代码调优）。

随机梯度下降的缺点包括：
    SGD 需要许多（参数），如 regularization parameter ( 正则化参数 ) 和迭代次数。
    SGD 对特征尺度敏感。

复杂度：
    O(knp):k,迭代次数，矩阵X(n,p)

技巧：
    对数据敏感：
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)  # Don't cheat - fit only on training data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)  # apply same transformation to test data


"""
from sklearn.linear_model import SGDClassifier


def SGD_fit():
    X = [[0., 0.], [1., 1.]]
    y = [0, 1]
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    clf.fit(X, y)
    print(clf.predict([[2., 2.]]))
    print(clf.coef_)
    print(clf.intercept_)


def plot_sgd_separating_hyperplane():
    """
    =========================================
    SGD: Maximum margin separating hyperplane
    =========================================

    Plot the maximum margin separating hyperplane within a two-class
    separable dataset using a linear Support Vector Machines classifier
    trained using SGD.
    """
    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import SGDClassifier
    from sklearn.datasets.samples_generator import make_blobs

    # we create 50 separable points
    X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

    # fit the model
    clf = SGDClassifier(loss="hinge", alpha=0.01, max_iter=200, fit_intercept=True)
    clf.fit(X, Y)

    # plot the line, the points, and the nearest vectors to the plane
    xx = np.linspace(-1, 5, 10)
    yy = np.linspace(-1, 5, 10)

    X1, X2 = np.meshgrid(xx, yy)
    Z = np.empty(X1.shape)
    for (i, j), val in np.ndenumerate(X1):
        x1 = val
        x2 = X2[i, j]
        p = clf.decision_function([[x1, x2]])
        Z[i, j] = p[0]

    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) 用这句更简单
    levels = [-1.0, 0.0, 1.0]
    linestyles = ['dashed', 'solid', 'dashed']
    colors = 'k'
    plt.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired,
                edgecolor='black', s=20)

    plt.axis('tight')
    plt.show()


def plot_sgd_iris():
    """
    ========================================
    Plot multi-class SGD on the iris dataset
    ========================================

    Plot decision surface of multi-class SGD on iris dataset.
    The hyperplanes corresponding to the three one-versus-all (OVA) classifiers
    are represented by the dashed lines.

    """
    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.linear_model import SGDClassifier

    # import some data to play with
    iris = datasets.load_iris()

    # we only take the first two features. We could
    # avoid this ugly slicing by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target
    colors = "bry"

    # shuffle
    idx = np.arange(X.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    h = .02  # step size in the mesh

    clf = SGDClassifier(alpha=0.001, max_iter=100).fit(X, y)

    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis('tight')

    # Plot also the training points
    for i, color in zip(clf.classes_, colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.Paired, edgecolor='black', s=20)
    plt.title("Decision surface of multi-class SGD")
    plt.axis('tight')

    # Plot the three one-against-all classifiers
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    coef = clf.coef_
    intercept = clf.intercept_

    def plot_hyperplane(c, _color):
        def line(x0):
            return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

        plt.plot([xmin, xmax], [line(xmin), line(xmax)],
                 ls="--", color=_color)

    for i, color in zip(clf.classes_, colors):
        plot_hyperplane(i, color)
    plt.legend()
    plt.show()


def plot_sgd_weighted_samples():
    """
    =====================
    SGD: Weighted samples
    =====================

    Plot decision function of a weighted dataset, where the size of points
    is proportional to its weight.
    """
    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import linear_model

    # we create 20 points
    np.random.seed(0)
    X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]
    y = [1] * 10 + [-1] * 10
    sample_weight = 100 * np.abs(np.random.randn(20))
    # and assign a bigger weight to the last 10 samples
    sample_weight[:10] *= 10

    # plot the weighted data points
    xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=sample_weight, alpha=0.9,
                cmap=plt.cm.bone, edgecolor='black')

    # fit the unweighted model
    clf = linear_model.SGDClassifier(alpha=0.01, max_iter=100)
    clf.fit(X, y)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    no_weights = plt.contour(xx, yy, Z, levels=[0], linestyles=['solid'])

    # fit the weighted model
    clf = linear_model.SGDClassifier(alpha=0.01, max_iter=100)
    clf.fit(X, y, sample_weight=sample_weight)
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    samples_weights = plt.contour(xx, yy, Z, levels=[0], linestyles=['dashed'])

    plt.legend([no_weights.collections[0], samples_weights.collections[0]],
               ["no weights", "with weights"], loc="lower left")

    plt.xticks(())
    plt.yticks(())
    plt.show()


def plot_sgd_comparison():
    """
    ==================================
    Comparing various online solvers
    ==================================

    An example showing how different online solvers perform
    on the hand-written digits dataset.

    """

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import SGDClassifier, Perceptron
    from sklearn.linear_model import PassiveAggressiveClassifier
    from sklearn.linear_model import LogisticRegression

    heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
    rounds = 20
    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    classifiers = [
        ("SGD", SGDClassifier(max_iter=100)),
        ("ASGD", SGDClassifier(average=True, max_iter=100)),
        ("Perceptron", Perceptron(tol=1e-3)),
        ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',
                                                             C=1.0, tol=1e-4)),
        ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                              C=1.0, tol=1e-4)),
        ("SAG", LogisticRegression(solver='sag', tol=1e-1, C=1.e4 / X.shape[0]))
    ]

    xx = 1. - np.array(heldout)

    for name, clf in classifiers:
        print("training %s" % name)
        rng = np.random.RandomState(42)
        yy = []
        for i in heldout:
            yy_ = []
            for r in range(rounds):
                X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, test_size=i, random_state=rng)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                yy_.append(1 - np.mean(y_pred == y_test))
            yy.append(np.mean(yy_))
        plt.plot(xx, yy, label=name)

    plt.legend(loc="upper right")
    plt.xlabel("Proportion train")
    plt.ylabel("Test Error Rate")
    plt.show()


if __name__ == '__main__':
    print(__doc__)
    plot_sgd_separating_hyperplane()
