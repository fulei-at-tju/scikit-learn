"""
集成方法
多标签多分类
特征选择
标签传播
神经网络
"""


def example():
    """
    ========
    集成方法
    ========
    """
    # bagging 方法
    # 使用均匀取样，每个样例的权重相等 平均预测
    from sklearn.ensemble import BaggingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    bagging = BaggingClassifier(KNeighborsClassifier(),
                                max_samples=0.5, max_features=0.5)

    # 随机森林 (bagging + dt 随机抽样训练样本)
    from sklearn.ensemble import RandomForestClassifier
    X = [[0, 0], [1, 1]]
    Y = [0, 1]
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(X, Y)

    # 极端随机森林（每次利用全部样本，训练，但分叉属性的划分值完全随机进行左右分叉）
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
    print(scores.mean())

    # #######################################################
    # boosting 提升方法（通常弱分类器组合）（不可并行）
    # 选部分数据作为第一次训练集，分错样本+剩余训练数据作为下一次训练集，循环，分类好的分类器权重大

    # AdaBoost
    from sklearn.model_selection import cross_val_score
    from sklearn.datasets import load_iris
    from sklearn.ensemble import AdaBoostClassifier

    iris = load_iris()
    clf = AdaBoostClassifier(n_estimators=100)
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    print(scores.mean())

    # GradientBoosting 梯度提升（通常弱分类器组合）
    # 样例：https://www.cnblogs.com/peizhe123/p/5086128.html
    from sklearn.datasets import make_hastie_10_2
    from sklearn.ensemble import GradientBoostingClassifier

    X, y = make_hastie_10_2(random_state=0)
    X_train, X_test = X[:2000], X[2000:]
    y_train, y_test = y[:2000], y[2000:]

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                     max_depth=1, random_state=0).fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    """
    =============
    多标签、多分类
    =============
    """
    # 多标签形式
    from sklearn.preprocessing import MultiLabelBinarizer
    y = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]
    print(MultiLabelBinarizer().fit_transform(y))

    # 利用ovr,ovo多分类
    from sklearn import datasets
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.multiclass import OneVsOneClassifier
    from sklearn.svm import LinearSVC
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf_ovr = OneVsRestClassifier(LinearSVC())
    print(clf_ovr.fit(X, y).predict(X))
    # 利用ovr可以进行多标签预测
    # also supports multilabel classification. To use this feature,
    # feed the classifier an indicator matrix, in which cell [i, j] indicates the presence of label j in sample i

    clf_ovo = OneVsOneClassifier(LinearSVC())
    print(clf_ovo.fit(X, y).predict(X))

    """
    =============
    特征选择
    =============
    """

    # 1.方差移除
    from sklearn.feature_selection import VarianceThreshold
    X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    print(sel.fit_transform(X))

    # 2.单变量特征选择
    #   SelectKBest
    # 	SelectPercentile
    # 	SelectFpr, SelectFdr, SelectFwe
    from sklearn.datasets import load_iris
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    iris = load_iris()
    X, y = iris.data, iris.target
    print(X.shape)

    X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
    print(X_new.shape)

    # scores_和pvalues_
    # p值越小，拒绝原假设，原假设：该特征和y不相关
    skb = SelectKBest(chi2, k=2).fit(X, y)
    print(skb.scores_)
    print(skb.pvalues_)

    # source function
    """
    For regression: f_regression, mutual_info_regression
    For classification: chi2, f_classif, mutual_info_classif
    """

    # 3.递归特征消除
    from sklearn.svm import SVC
    from sklearn.datasets import load_digits
    from sklearn.feature_selection import RFE
    import matplotlib.pyplot as plt

    # Load the digits dataset
    digits = load_digits()
    X = digits.images.reshape((len(digits.images), -1))
    y = digits.target

    # Create the RFE object and rank each pixel
    svc = SVC(kernel="linear", C=1)
    rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
    rfe.fit(X, y)
    ranking = rfe.ranking_.reshape(digits.images[0].shape)

    # Plot pixel ranking
    plt.matshow(ranking, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title("Ranking of pixels with RFE")
    plt.show()

    # 4.1 SelectFromModel
    from sklearn.svm import LinearSVC
    from sklearn.datasets import load_iris
    from sklearn.feature_selection import SelectFromModel
    iris = load_iris()
    X, y = iris.data, iris.target
    print(X.shape)

    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(X)
    print(X_new.shape)

    # 4.2 树结构 feature_importances_
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.datasets import load_iris
    from sklearn.feature_selection import SelectFromModel
    iris = load_iris()
    X, y = iris.data, iris.target
    print(X.shape)

    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X, y)
    print(clf.feature_importances_)

    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print(X_new.shape)

    """
    =============
    神经网络
    =============
    """
    # 神经网络，参数讲解
    from sklearn.neural_network import MLPClassifier
    X = [[0., 0.], [1., 1.]]
    y = [0, 1]
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)

    print(clf.fit(X, y))
    print(clf.predict([[2., 2.], [-1., -2.]]))
    print(clf.coefs_)
    print(clf.intercepts_)
    print(clf.loss_)


def plot_bias_variance():
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


def plot_label_propagation_structure():
    # Authors: Clay Woolam <clay@woolam.org>
    #          Andreas Mueller <amueller@ais.uni-bonn.de>
    # License: BSD

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.semi_supervised import label_propagation
    from sklearn.datasets import make_circles

    # generate ring with inner box
    n_samples = 200
    X, y = make_circles(n_samples=n_samples, shuffle=False)
    outer, inner = 0, 1
    labels = np.full(n_samples, -1.)
    labels[0] = outer
    labels[-1] = inner

    # #############################################################################
    # Learn with LabelSpreading
    # kernel在代码中是计算ap距离矩阵的方法
    label_spread = label_propagation.LabelSpreading(kernel='knn', alpha=0.8)
    label_spread.fit(X, labels)

    # #############################################################################
    # Plot output labels
    output_labels = label_spread.transduction_
    plt.figure(figsize=(8.5, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[labels == outer, 0], X[labels == outer, 1], color='navy',
                marker='s', lw=0, label="outer labeled", s=10)
    plt.scatter(X[labels == inner, 0], X[labels == inner, 1], color='c',
                marker='s', lw=0, label='inner labeled', s=10)
    plt.scatter(X[labels == -1, 0], X[labels == -1, 1], color='darkorange',
                marker='.', label='unlabeled')
    plt.legend(scatterpoints=1, shadow=False, loc='upper right')
    plt.title("Raw data (2 classes=outer and inner)")

    plt.subplot(1, 2, 2)
    output_label_array = np.asarray(output_labels)
    outer_numbers = np.where(output_label_array == outer)[0]
    inner_numbers = np.where(output_label_array == inner)[0]
    plt.scatter(X[outer_numbers, 0], X[outer_numbers, 1], color='navy',
                marker='s', lw=0, s=10, label="outer learned")
    plt.scatter(X[inner_numbers, 0], X[inner_numbers, 1], color='c',
                marker='s', lw=0, s=10, label="inner learned")
    plt.legend(scatterpoints=1, shadow=False, loc='upper right')
    plt.title("Labels learned with Label Spreading (KNN)")

    plt.subplots_adjust(left=0.07, bottom=0.07, right=0.93, top=0.92)
    plt.show()


def plot_mlp_alpha():
    """
    直观观察正则项系数改变如何影响分类器
    hidden_layer_sizes=(30,30,30)
    alpha越小，正则项影响越小，分类越容易过拟合
    """
    print(__doc__)

    # Author: Issam H. Laradji
    # License: BSD 3 clause

    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_moons, make_circles, make_classification
    from sklearn.neural_network import MLPClassifier

    h = .02  # step size in the mesh

    alphas = np.logspace(-5, 3, 5)
    names = []
    for i in alphas:
        names.append('alpha ' + str(i))

    classifiers = []
    for i in alphas:
        classifiers.append(MLPClassifier(alpha=i, random_state=1))

    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=0, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [make_moons(noise=0.3, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1),
                linearly_separable]

    figure = plt.figure(figsize=(17, 9))
    i = 1
    # iterate over datasets
    for X, y in datasets:
        # preprocess dataset, split into training and test part
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                       edgecolors='black', s=25)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       alpha=0.6, edgecolors='black', s=25)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1

    figure.subplots_adjust(left=.02, right=.98)
    plt.show()


if __name__ == '__main__':
    # 1、集成方法各个方法总结
    example()

    # 2、集成方法bagging与非集成对比减少方差
    plot_bias_variance()

    # 3、多标签、多分类
    example()

    # 4、特征选择
    example()

    # 5、半监督学习-标签传播算法
    plot_label_propagation_structure()

    # 6、神经网络
    example()

    # 6.1、神经网络 正则项系数影响
    plot_mlp_alpha()
