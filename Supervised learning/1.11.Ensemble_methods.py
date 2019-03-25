"""
==========================
Ensemble methods（集成方法）
==========================

可分类：Classifier
可回归：Regressor
可特征选择（排序）：feature_importances_

正则化Regularization（优化）：
    Shrinkage：learning_rate
    Subsampling：subsample

https://scikit-learn.org/stable/modules/ensemble.html

将已有的分类或回归算法通过一定方式组合起来，形成一个性能更加强大的分类器，更准确的说这是一种分类算法的组装方法。
以便通过单个估计器来提高泛化/鲁棒性。

两类集成方法：
    1.averaging methods 平均方法（bagging）（通常强分类器组合）（可并行n_jobs）（减小方差）
        独立（原始样本中抽取训练集）构建分类器，然后平均其预测。平均来说，组合估计器通常比单个基准估计器更好，因为它的方差减小。
        样本选择：训练集是在原始集中有放回选取的，从原始集中选出的各轮训练集之间是独立的。
        样例权重：使用均匀取样，每个样例的权重相等
        Bagging + 决策树 = 随机森林

    2.boosting methods 提升方法(boosting)（通常弱分类器组合）（不可并行）
        依次构建基本估计量，并尝试减少组合估计量的偏差。若干弱分类器的集合。
        样本选择：每一轮的训练集不变，只是训练集中每个样例在分类器中的权重发生变化。而权值是根据上一轮的分类结果进行调整。
        样例权重：根据错误率不断调整样例的权值，错误率越大则权重越大。
        boosting: 选部分数据作为第一次训练集，分错样本+剩余训练数据作为下一次训练集，循环，分类好的分类器权重大
        adaboost:将错误样本的比重调整，作为每次训练数据，平均投票
        AdaBoost + 决策树 = 提升树
        Gradient Boosting + 决策树 = GBDT

        2.1 Boosting 与 Gradient Boosting
            Boosting:
                是利用一些弱分类器的组合来构造一个强分类器,通过分错样本的权值改变
            Gradient Boosting（梯度提升）：
                每一次的计算是为了减少上一次的残差，就可以在残差减少的梯度方向上建立一个新的模型。
                也就是使得之前模型的残差往梯度方向减少

    3.Voting Classifier（Hard Voting）（Soft Voting）

"""


def plot_bias_variance():
    """
    ============================================================
    Single estimator versus bagging: bias-variance decomposition
    ============================================================

    This example illustrates and compares the bias-variance decomposition of the
    expected mean squared error of a single estimator against a bagging ensemble.

    In regression, the expected mean squared error of an estimator can be
    decomposed in terms of bias, variance and noise. On average over datasets of
    the regression problem, the bias term measures the average amount by which the
    predictions of the estimator differ from the predictions of the best possible
    estimator for the problem (i.e., the Bayes model). The variance term measures
    the variability of the predictions of the estimator when fit over different
    instances LS of the problem. Finally, the noise measures the irreducible part
    of the error which is due the variability in the data.

    The upper left figure illustrates the predictions (in dark red) of a single
    decision tree trained over a random dataset LS (the blue dots) of a toy 1d
    regression problem. It also illustrates the predictions (in light red) of other
    single decision trees trained over other (and different) randomly drawn
    instances LS of the problem. Intuitively, the variance term here corresponds to
    the width of the beam of predictions (in light red) of the individual
    estimators. The larger the variance, the more sensitive are the predictions for
    `x` to small changes in the training set. The bias term corresponds to the
    difference between the average prediction of the estimator (in cyan) and the
    best possible model (in dark blue). On this problem, we can thus observe that
    the bias is quite low (both the cyan and the blue curves are close to each
    other) while the variance is large (the red beam is rather wide).

    The lower left figure plots the pointwise decomposition of the expected mean
    squared error of a single decision tree. It confirms that the bias term (in
    blue) is low while the variance is large (in green). It also illustrates the
    noise part of the error which, as expected, appears to be constant and around
    `0.01`.

    The right figures correspond to the same plots but using instead a bagging
    ensemble of decision trees. In both figures, we can observe that the bias term
    is larger than in the previous case. In the upper right figure, the difference
    between the average prediction (in cyan) and the best possible model is larger
    (e.g., notice the offset around `x=2`). In the lower right figure, the bias
    curve is also slightly higher than in the lower left figure. In terms of
    variance however, the beam of predictions is narrower, which suggests that the
    variance is lower. Indeed, as the lower right figure confirms, the variance
    term (in green) is lower than for single decision trees. Overall, the bias-
    variance decomposition is therefore no longer the same. The tradeoff is better
    for bagging: averaging several decision trees fit on bootstrap copies of the
    dataset slightly increases the bias term but allows for a larger reduction of
    the variance, which results in a lower overall mean squared error (compare the
    red curves int the lower figures). The script output also confirms this
    intuition. The total error of the bagging ensemble is lower than the total
    error of a single decision tree, and this difference indeed mainly stems from a
    reduced variance.

    For further details on bias-variance decomposition, see section 7.3 of [1]_.

    References
    ----------

    .. [1] T. Hastie, R. Tibshirani and J. Friedman,
           "Elements of Statistical Learning", Springer, 2009.

    """
    print(__doc__)

    # Author: Gilles Louppe <g.louppe@gmail.com>
    # License: BSD 3 clause

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.ensemble import BaggingRegressor
    from sklearn.tree import DecisionTreeRegressor

    # Settings
    n_repeat = 50  # Number of iterations for computing expectations
    n_train = 50  # Size of the training set
    n_test = 1000  # Size of the test set
    noise = 0.1  # Standard deviation of the noise
    np.random.seed(0)

    # Change this for exploring the bias-variance decomposition of other
    # estimators. This should work well for estimators with high variance (e.g.,
    # decision trees or KNN), but poorly for estimators with low variance (e.g.,
    # linear models).
    estimators = [("Tree", DecisionTreeRegressor()),
                  ("Bagging(Tree)", BaggingRegressor(DecisionTreeRegressor()))]

    n_estimators = len(estimators)

    # Generate data
    def f(x):
        x = x.ravel()

        return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)

    def generate(n_samples, _noise, _n_repeat=1):
        _X = np.random.rand(n_samples) * 10 - 5
        _X = np.sort(_X)

        if _n_repeat == 1:
            _y = f(_X) + np.random.normal(0.0, _noise, n_samples)
        else:
            _y = np.zeros((n_samples, _n_repeat))

            for _i in range(_n_repeat):
                _y[:, _i] = f(_X) + np.random.normal(0.0, _noise, n_samples)

        _X = _X.reshape((n_samples, 1))

        return _X, _y

    X_train = []
    y_train = []

    for i in range(n_repeat):
        X, y = generate(n_samples=n_train, _noise=noise)
        X_train.append(X)
        y_train.append(y)

    X_test, y_test = generate(n_samples=n_test, _noise=noise, _n_repeat=n_repeat)

    plt.figure(figsize=(10, 8))

    # Loop over estimators to compare
    for n, (name, estimator) in enumerate(estimators):
        # Compute predictions
        y_predict = np.zeros((n_test, n_repeat))

        for i in range(n_repeat):
            estimator.fit(X_train[i], y_train[i])
            y_predict[:, i] = estimator.predict(X_test)

        # Bias^2 + Variance + Noise decomposition of the mean squared error
        y_error = np.zeros(n_test)

        for i in range(n_repeat):
            for j in range(n_repeat):
                y_error += (y_test[:, j] - y_predict[:, i]) ** 2

        y_error /= (n_repeat * n_repeat)

        y_noise = np.var(y_test, axis=1)
        y_bias = (f(X_test) - np.mean(y_predict, axis=1)) ** 2
        y_var = np.var(y_predict, axis=1)

        print("{0}: {1:.4f} (error) = {2:.4f} (bias^2) "
              " + {3:.4f} (var) + {4:.4f} (noise)".format(name,
                                                          np.mean(y_error),
                                                          np.mean(y_bias),
                                                          np.mean(y_var),
                                                          np.mean(y_noise)))

        # Plot figures
        plt.subplot(2, n_estimators, n + 1)
        plt.plot(X_test, f(X_test), "b", label="$f(x)$")
        plt.plot(X_train[0], y_train[0], ".b", label="LS ~ $y = f(x)+noise$")

        for i in range(n_repeat):
            if i == 0:
                plt.plot(X_test, y_predict[:, i], "r", label="$\^y(x)$")
            else:
                plt.plot(X_test, y_predict[:, i], "r", alpha=0.05)

        plt.plot(X_test, np.mean(y_predict, axis=1), "c",
                 label="$\mathbb{E}_{LS} \^y(x)$")

        plt.xlim([-5, 5])
        plt.title(name)

        if n == n_estimators - 1:
            plt.legend(loc=(1.1, .5))

        plt.subplot(2, n_estimators, n_estimators + n + 1)
        plt.plot(X_test, y_error, "r", label="$error(x)$")
        plt.plot(X_test, y_bias, "b", label="$bias^2(x)$"),
        plt.plot(X_test, y_var, "g", label="$variance(x)$"),
        plt.plot(X_test, y_noise, "c", label="$noise(x)$")

        plt.xlim([-5, 5])
        plt.ylim([0, 0.1])

        if n == n_estimators - 1:
            plt.legend(loc=(1.1, .5))

    plt.subplots_adjust(right=.75)
    plt.show()


def different_tree():
    """
    随机森林（RandomForestClassifier）
        bagging方式随机挑选子训练集，训练子树
    极端随机树（ExtraTreesClassifier）
        每次利用全部样本，训练，但分叉属性的划分值完全随机进行左右分叉
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import make_blobs
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier

    X, y = make_blobs(n_samples=10000, n_features=10, centers=100,
                      random_state=0)

    clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
                                 random_state=0)
    scores = cross_val_score(clf, X, y, cv=5)
    print(scores.mean())

    clf = RandomForestClassifier(n_estimators=10, max_depth=None,
                                 min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, X, y, cv=5)
    print(scores.mean())

    clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
                               min_samples_split=2, random_state=0)
    scores = cross_val_score(clf, X, y, cv=5)
    print(scores.mean() > 0.999)


def plot_forest_importances_faces():
    """
    =================================================
    Pixel importances with a parallel forest of trees
    =================================================

    This example shows the use of forests of trees to evaluate the importance
    of the pixels in an image classification task (faces). The hotter the pixel,
    the more important.

    The code below also illustrates how the construction and the computation
    of the predictions can be parallelized within multiple jobs.
    """
    print(__doc__)

    from time import time
    import matplotlib.pyplot as plt

    from sklearn.datasets import fetch_olivetti_faces
    from sklearn.ensemble import ExtraTreesClassifier

    # Number of cores to use to perform parallel fitting of the forest model
    n_jobs = 1

    # Load the faces dataset
    data = fetch_olivetti_faces()
    X = data.images.reshape((len(data.images), -1))
    y = data.target

    mask = y < 5  # Limit to 5 classes
    X = X[mask]
    y = y[mask]

    # Build a forest and compute the pixel importances
    print("Fitting ExtraTreesClassifier on faces data with %d cores..." % n_jobs)
    t0 = time()
    forest = ExtraTreesClassifier(n_estimators=1000,
                                  max_features=128,
                                  n_jobs=n_jobs,
                                  random_state=0)

    forest.fit(X, y)
    print("done in %0.3fs" % (time() - t0))
    importances = forest.feature_importances_
    importances = importances.reshape(data.images[0].shape)

    # Plot pixel importances
    plt.matshow(importances, cmap=plt.cm.hot)
    plt.title("Pixel importances with forests of trees")
    plt.show()


def plot_multioutput_face_completion():
    """
    ==============================================
    Face completion with a multi-output estimators
    ==============================================

    This example shows the use of multi-output estimator to complete images.
    The goal is to predict the lower half of a face given its upper half.

    The first column of images shows true faces. The next columns illustrate
    how extremely randomized trees, k nearest neighbors, linear
    regression and ridge regression complete the lower half of those faces.

    """
    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import fetch_olivetti_faces
    from sklearn.utils.validation import check_random_state

    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import RidgeCV

    # Load the faces datasets
    data = fetch_olivetti_faces()
    targets = data.target

    data = data.images.reshape((len(data.images), -1))
    train = data[targets < 30]
    test = data[targets >= 30]  # Test on independent people

    # Test on a subset of people
    n_faces = 5
    rng = check_random_state(4)
    face_ids = rng.randint(test.shape[0], size=(n_faces,))
    test = test[face_ids, :]

    n_pixels = data.shape[1]
    # Upper half of the faces
    X_train = train[:, :(n_pixels + 1) // 2]
    # Lower half of the faces
    y_train = train[:, n_pixels // 2:]
    X_test = test[:, :(n_pixels + 1) // 2]
    y_test = test[:, n_pixels // 2:]

    # Fit estimators
    ESTIMATORS = {
        "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32,
                                           random_state=0),
        "K-nn": KNeighborsRegressor(),
        "Linear regression": LinearRegression(),
        "Ridge": RidgeCV(),
    }

    y_test_predict = dict()
    for name, estimator in ESTIMATORS.items():
        estimator.fit(X_train, y_train)
        y_test_predict[name] = estimator.predict(X_test)

    # Plot the completed faces
    image_shape = (64, 64)

    n_cols = 1 + len(ESTIMATORS)
    plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
    plt.suptitle("Face completion with multi-output estimators", size=16)

    for i in range(n_faces):
        true_face = np.hstack((X_test[i], y_test[i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 1,
                              title="true faces")

        sub.axis("off")
        sub.imshow(true_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")

        for j, est in enumerate(sorted(ESTIMATORS)):
            completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

            if i:
                sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)

            else:
                sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j,
                                  title=est)

            sub.axis("off")
            sub.imshow(completed_face.reshape(image_shape),
                       cmap=plt.cm.gray,
                       interpolation="nearest")

    plt.show()


def AdaBoost():
    """
    1. 先通过对N个训练样本的学习得到第一个弱分类器；
    2. 将分错的样本和其他的新数据一起构成一个新的N个的训练样本，通过对这个样本的学习得到第二个弱分类器 ；
    3. 将1和2都分错了的样本加上其他的新样本构成另一个新的N个的训练样本，通过对这个样本的学习得到第三个弱分类器；
    4. 最终经过提升的强分类器。即某个数据被分为哪一类要由各分类器权值决定。
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_iris
    from sklearn.ensemble import AdaBoostClassifier

    iris = load_iris()
    clf = AdaBoostClassifier(n_estimators=100)
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    print(scores.mean())


def plot_adaboost_hastie_10_2():
    """
    =============================
    Discrete versus Real AdaBoost
    =============================

    This example is based on Figure 10.2 from Hastie et al 2009 [1]_ and
    illustrates the difference in performance between the discrete SAMME [2]_
    boosting algorithm and real SAMME.R boosting algorithm. Both algorithms are
    evaluated on a binary classification task where the target Y is a non-linear
    function of 10 input features.

    Discrete SAMME AdaBoost adapts based on errors in predicted class labels
    whereas real SAMME.R uses the predicted class probabilities.

    .. [1] T. Hastie, R. Tibshirani and J. Friedman, "Elements of Statistical
        Learning Ed. 2", Springer, 2009.

    .. [2] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    """
    print(__doc__)

    # Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>,
    #         Noel Dawe <noel.dawe@gmail.com>
    #
    # License: BSD 3 clause

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import zero_one_loss
    from sklearn.ensemble import AdaBoostClassifier

    n_estimators = 400
    # A learning rate of 1. may not be optimal for both SAMME and SAMME.R
    learning_rate = 1.

    X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)

    X_test, y_test = X[2000:], y[2000:]
    X_train, y_train = X[:2000], y[:2000]

    dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
    dt_stump.fit(X_train, y_train)
    dt_stump_err = 1.0 - dt_stump.score(X_test, y_test)

    dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
    dt.fit(X_train, y_train)
    dt_err = 1.0 - dt.score(X_test, y_test)

    ada_discrete = AdaBoostClassifier(
        base_estimator=dt_stump,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        algorithm="SAMME")
    ada_discrete.fit(X_train, y_train)

    ada_real = AdaBoostClassifier(
        base_estimator=dt_stump,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        algorithm="SAMME.R")
    ada_real.fit(X_train, y_train)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot([1, n_estimators], [dt_stump_err] * 2, 'k-',
            label='Decision Stump Error')
    ax.plot([1, n_estimators], [dt_err] * 2, 'k--',
            label='Decision Tree Error')

    ada_discrete_err = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ada_discrete.staged_predict(X_test)):
        ada_discrete_err[i] = zero_one_loss(y_pred, y_test)

    ada_discrete_err_train = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ada_discrete.staged_predict(X_train)):
        ada_discrete_err_train[i] = zero_one_loss(y_pred, y_train)

    ada_real_err = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ada_real.staged_predict(X_test)):
        ada_real_err[i] = zero_one_loss(y_pred, y_test)

    ada_real_err_train = np.zeros((n_estimators,))
    for i, y_pred in enumerate(ada_real.staged_predict(X_train)):
        ada_real_err_train[i] = zero_one_loss(y_pred, y_train)

    ax.plot(np.arange(n_estimators) + 1, ada_discrete_err,
            label='Discrete AdaBoost Test Error',
            color='red')
    ax.plot(np.arange(n_estimators) + 1, ada_discrete_err_train,
            label='Discrete AdaBoost Train Error',
            color='blue')
    ax.plot(np.arange(n_estimators) + 1, ada_real_err,
            label='Real AdaBoost Test Error',
            color='orange')
    ax.plot(np.arange(n_estimators) + 1, ada_real_err_train,
            label='Real AdaBoost Train Error',
            color='green')

    ax.set_ylim((0.0, 0.5))
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('error rate')

    leg = ax.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.7)

    plt.show()


def plot_adaboost_regression():
    """
    ======================================
    Decision Tree Regression with AdaBoost
    ======================================

    A decision tree is boosted using the AdaBoost.R2 [1]_ algorithm on a 1D
    sinusoidal dataset with a small amount of Gaussian noise.
    299 boosts (300 decision trees) is compared with a single decision tree
    regressor. As the number of boosts is increased the regressor can fit more
    detail.

    .. [1] H. Drucker, "Improving Regressors using Boosting Techniques", 1997.

    """
    print(__doc__)

    # Author: Noel Dawe <noel.dawe@gmail.com>
    #
    # License: BSD 3 clause

    # importing necessary libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import AdaBoostRegressor

    # Create the dataset
    rng = np.random.RandomState(1)
    X = np.linspace(0, 6, 100)[:, np.newaxis]
    y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

    # Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=4)

    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                               n_estimators=300, random_state=rng)

    regr_1.fit(X, y)
    regr_2.fit(X, y)

    # Predict
    y_1 = regr_1.predict(X)
    y_2 = regr_2.predict(X)

    # Plot the results
    plt.figure()
    plt.scatter(X, y, c="k", label="training samples")
    plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
    plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Boosted Decision Tree Regression")
    plt.legend()
    plt.show()


def plot_gradient_boosting_regression():
    """
    ============================
    Gradient Boosting regression
    ============================

    Demonstrate Gradient Boosting on the Boston housing dataset.

    This example fits a Gradient Boosting model with least squares loss and
    500 regression trees of depth 4.
    """
    print(__doc__)

    # Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
    #
    # License: BSD 3 clause

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import ensemble
    from sklearn import datasets
    from sklearn.utils import shuffle
    from sklearn.metrics import mean_squared_error

    # #############################################################################
    # Load data
    boston = datasets.load_boston()
    X, y = shuffle(boston.data, boston.target, random_state=13)
    X = X.astype(np.float32)
    offset = int(X.shape[0] * 0.9)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    # #############################################################################
    # Fit regression model
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)

    clf.fit(X_train, y_train)
    mse = mean_squared_error(y_test, clf.predict(X_test))
    print("MSE: %.4f" % mse)

    # #############################################################################
    # Plot training deviance

    # compute test set deviance
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_predict(X_test)):
        test_score[i] = clf.loss_(y_test, y_pred)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')

    # #############################################################################
    # Plot feature importance
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, boston.feature_names[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


def Weighted_Average_Probabilities():
    """
      (Soft Voting)
    """
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import VotingClassifier

    # Loading some example data
    iris = datasets.load_iris()
    X = iris.data[:, [0, 2]]
    y = iris.target

    # Training classifiers
    clf1 = DecisionTreeClassifier(max_depth=4)
    clf2 = KNeighborsClassifier(n_neighbors=7)
    clf3 = SVC(gamma='scale', kernel='rbf', probability=True)
    eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)],
                            voting='soft', weights=[2, 1, 2])

    clf1 = clf1.fit(X, y)
    clf2 = clf2.fit(X, y)
    clf3 = clf3.fit(X, y)
    eclf = eclf.fit(X, y)

    for clf, label in zip([clf1, clf2, clf3, eclf],
                          ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


if __name__ == '__main__':
    plot_bias_variance()
