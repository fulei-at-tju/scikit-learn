"""
============================
Feature selection（特征选择）
============================

特征选择/降维，以提高估计器的准确度分数或提高其在高维数据集上的性能。

1.删除低方差特征：
    方差为0，特征属性都一至，无可区分性,一般阈值.8 * (1 - .8)

2.单变量特征选择:
    通过选择基于统计测试的最佳特征进行单变量特征选择
    SelectKBest removes all but the  highest scoring features
    SelectPercentile removes all but a user-specified highest scoring percentage of features
    using common univariate statistical tests for each feature:
        false positive rate SelectFpr,
        false discovery rate SelectFdr,
        family wise error SelectFwe.
    GenericUnivariateSelect allows to perform univariate feature selection with a configurable strategy. This allows to select the best univariate selection strategy with hyper-parameter search estimator.

    SelectKBest和SelectPercentile的评分函数：
        For regression: f_regression, mutual_info_regression
        For classification: chi2(卡方分布), f_classif, mutual_info_classif

3.递归特征消除（rfe, Recursive feature elimination）
4.从模型中选择(SelectFromModel)
    线性模型中，特征系数为0
    树形结构

    threshold : string, float, optional default None
        The threshold value to use for feature selection. Features whose
        importance is greater or equal are kept while the others are
        discarded. If "median" (resp. "mean"), then the ``threshold`` value is
        the median (resp. the mean) of the feature importances. A scaling
        factor (e.g., "1.25*mean") may also be used. If None and if the
        estimator has a parameter penalty set to l1, either explicitly
        or implicitly (e.g, Lasso), the threshold used is 1e-5.
        Otherwise, "mean" is used by default.


"""


def Removing_features_with_low_variance():
    from sklearn.feature_selection import VarianceThreshold
    X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    print(sel.fit_transform(X))


def Univariate_feature_selection():
    """
    For regression: f_regression, mutual_info_regression
    For classification: chi2, f_classif, mutual_info_classif
    """
    from sklearn.datasets import load_iris
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    iris = load_iris()
    X, y = iris.data, iris.target
    print(X.shape)

    X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
    print(X_new.shape)


def plot_rfe_digits():
    """
    =============================
    Recursive feature elimination
    =============================

    A recursive feature elimination example showing the relevance of pixels in
    a digit classification task.

    .. note::

        See also :ref:`sphx_glr_auto_examples_feature_selection_plot_rfe_with_cross_validation.py`

    """
    print(__doc__)

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


def plot_rfe_with_cross_validation():
    """
    ===================================================
    Recursive feature elimination with cross-validation
    ===================================================

    A recursive feature elimination example with automatic tuning of the
    number of features selected with cross-validation.
    """
    print(__doc__)

    import matplotlib.pyplot as plt
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    from sklearn.feature_selection import RFECV
    from sklearn.datasets import make_classification

    # Build a classification task using 3 informative features
    X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                               n_redundant=2, n_repeated=0, n_classes=8,
                               n_clusters_per_class=1, random_state=0)

    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring='accuracy')
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()


def L1_based_feature_selection():
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


def Tree_based_feature_selection():
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


if __name__ == '__main__':
    Tree_based_feature_selection()
