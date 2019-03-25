"""
==================================================
Novelty and Outlier Detection (新奇点和异常点检测）
==================================================

outlier detection（无监督）:
    当训练数据中包含离群点，模型训练时要匹配训练数据的中心样本，忽视训练样本中的其它异常点
 	The training data contains outliers which are defined as observations that
 	 are far from the others. Outlier detection estimators thus try to fit the
 	  regions where the training data is the most concentrated, ignoring the
 	  deviant observations.

novelty detection（半监督）:
    当训练数据中没有离群点，我们的目标是用训练好的模型去检测另外新发现的样本
 	The training data is not polluted by outliers and we are interested in
 	detecting whether a new observation is an outlier. In this context an
 	outlier is also called a novelty.

Inliers are labeled 1, while outliers are labeled -1

Robust covariance: 根据矩阵形成一个椭圆区域

One-Class SVM：单分类问题，训练数据只有一类。***多用于 Novelty Detection

Isolation Forest（孤立森林）：随机对数据进行切割划分，所有数据都划分为独立点。密度越大的数据
    划分次数越多，而密度低的数据划分次数越低。所以深度较低的叶子节点，属于异常点

Local Outlier Factor：一个样本点周围的样本点所处位置的平均密度比上该样本点所在位置的密度
    K-邻近距离（k-distance）、局部可达密度、局部异常因子（local outlier factor）
    如果数据点 p 的 LOF 得分在1附近，表明数据点p的局部密度跟它的邻居们差不多；
    如果数据点 p 的 LOF 得分小于1，表明数据点p处在一个相对密集的区域，不像是一个异常点；
    如果数据点 p 的 LOF 得分远大于1，表明数据点p跟其他点比较疏远，很有可能是一个异常点
    https://yq.aliyun.com/articles/152627?t=t1
"""


def plot_anomaly_comparison():
    """
    ============================================================================
    Comparing anomaly detection algorithms for outlier detection on toy datasets
    ============================================================================

    This example shows characteristics of different anomaly detection algorithms
    on 2D datasets. Datasets contain one or two modes (regions of high density)
    to illustrate the ability of algorithms to cope with multimodal data.

    For each dataset, 15% of samples are generated as random uniform noise. This
    proportion is the value given to the nu parameter of the OneClassSVM and the
    contamination parameter of the other outlier detection algorithms.
    Decision boundaries between inliers and outliers are displayed in black
    except for Local Outlier Factor (LOF) as it has no predict method to be applied
    on new data when it is used for outlier detection.

    The :class:`svm.OneClassSVM` is known to be sensitive to outliers and thus does
    not perform very well for outlier detection. This estimator is best suited for
    novelty detection when the training set is not contaminated by outliers.
    That said, outlier detection in high-dimension, or without any assumptions on
    the distribution of the inlying data is very challenging, and a One-class SVM
    might give useful results in these situations depending on the value of its
    hyperparameters.

    :class:`covariance.EllipticEnvelope` assumes the data is Gaussian and learns
    an ellipse. It thus degrades when the data is not unimodal. Notice however
    that this estimator is robust to outliers.

    :class:`ensemble.IsolationForest` and :class:`neighbors.LocalOutlierFactor`
    seem to perform reasonably well for multi-modal data sets. The advantage of
    :class:`neighbors.LocalOutlierFactor` over the other estimators is shown for
    the third data set, where the two modes have different densities. This
    advantage is explained by the local aspect of LOF, meaning that it only
    compares the score of abnormality of one sample with the scores of its
    neighbors.

    Finally, for the last data set, it is hard to say that one sample is more
    abnormal than another sample as they are uniformly distributed in a
    hypercube. Except for the :class:`svm.OneClassSVM` which overfits a little, all
    estimators present decent solutions for this situation. In such a case, it
    would be wise to look more closely at the scores of abnormality of the samples
    as a good estimator should assign similar scores to all the samples.

    While these examples give some intuition about the algorithms, this
    intuition might not apply to very high dimensional data.

    Finally, note that parameters of the models have been here handpicked but
    that in practice they need to be adjusted. In the absence of labelled data,
    the problem is completely unsupervised so model selection can be a challenge.
    """

    # Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
    #         Albert Thomas <albert.thomas@telecom-paristech.fr>
    # License: BSD 3 clause

    import time

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    from sklearn import svm
    from sklearn.datasets import make_moons, make_blobs
    from sklearn.covariance import EllipticEnvelope
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    print(__doc__)

    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

    # Example settings
    n_samples = 300
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers

    # define outlier/anomaly detection methods to be compared
    anomaly_algorithms = [
        ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
        ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                          gamma=0.1)),
        ("Isolation Forest", IsolationForest(behaviour='new',
                                             contamination=outliers_fraction,
                                             random_state=42)),
        ("Local Outlier Factor", LocalOutlierFactor(
            n_neighbors=35, contamination=outliers_fraction))]

    # Define datasets
    blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
    datasets = [
        make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5,
                   **blobs_params)[0],
        make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5],
                   **blobs_params)[0],
        make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3],
                   **blobs_params)[0],
        4. * (make_moons(n_samples=n_samples, noise=.05, random_state=0)[0] -
              np.array([0.5, 0.25])),
        14. * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)]

    # Compare given classifiers under given settings
    xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
                         np.linspace(-7, 7, 150))

    plt.figure(figsize=(len(anomaly_algorithms) * 2 + 3, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)

    plot_num = 1
    rng = np.random.RandomState(42)

    for i_dataset, X in enumerate(datasets):
        # Add outliers
        X = np.concatenate([X, rng.uniform(low=-6, high=6,
                                           size=(n_outliers, 2))], axis=0)

        for name, algorithm in anomaly_algorithms:
            t0 = time.time()
            algorithm.fit(X)
            t1 = time.time()
            plt.subplot(len(datasets), len(anomaly_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)

            # fit the data and tag outliers
            if name == "Local Outlier Factor":
                y_pred = algorithm.fit_predict(X)
            else:
                y_pred = algorithm.fit(X).predict(X)

            # plot the levels lines and the points
            if name != "Local Outlier Factor":  # LOF does not implement predict
                Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

            colors = np.array(['#377eb8', '#ff7f00'])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])

            plt.xlim(-7, 7)
            plt.ylim(-7, 7)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1

    plt.show()


def plot_oneclass():
    """
    ==========================================
    One-class SVM with non-linear kernel (RBF)
    ==========================================

    An example using a one-class SVM for novelty detection.

    :ref:`One-class SVM <svm_outlier_detection>` is an unsupervised
    algorithm that learns a decision function for novelty detection:
    classifying new data as similar or different to the training set.
    """
    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.font_manager
    from sklearn import svm

    xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
    # Generate train data
    X = 0.3 * np.random.randn(100, 2)
    X_train = np.r_[X + 2, X - 2]
    # Generate some regular novel observations
    X = 0.3 * np.random.randn(20, 2)
    X_test = np.r_[X + 2, X - 2]
    # Generate some abnormal novel observations
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

    # fit the model
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)
    n_error_train = y_pred_train[y_pred_train == -1].size
    n_error_test = y_pred_test[y_pred_test == -1].size
    n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

    # plot the line, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("Novelty Detection")
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')

    s = 40
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=s, edgecolors='k')
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='blueviolet', s=s,
                     edgecolors='k')
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='gold', s=s,
                    edgecolors='k')
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.legend([a.collections[0], b1, b2, c],
               ["learned frontier", "training observations",
                "new regular observations", "new abnormal observations"],
               loc="upper left",
               prop=matplotlib.font_manager.FontProperties(size=11))
    plt.xlabel(
        "error train: %d/200 ; errors novel regular: %d/40 ; "
        "errors novel abnormal: %d/40"
        % (n_error_train, n_error_test, n_error_outliers))
    plt.show()


def plot_isolation_forest():
    """
    ==========================================
    IsolationForest example
    ==========================================

    An example using :class:`sklearn.ensemble.IsolationForest` for anomaly
    detection.

    The IsolationForest 'isolates' observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum
    values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the
    number of splittings required to isolate a sample is equivalent to the path
    length from the root node to the terminating node.

    This path length, averaged over a forest of such random trees, is a measure
    of normality and our decision function.

    Random partitioning produces noticeable shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path lengths
    for particular samples, they are highly likely to be anomalies.

    """
    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest

    rng = np.random.RandomState(42)

    # Generate train data
    X = 0.3 * rng.randn(100, 2)
    X_train = np.r_[X + 2, X - 2]
    # Generate some regular novel observations
    X = 0.3 * rng.randn(20, 2)
    X_test = np.r_[X + 2, X - 2]
    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

    # fit the model
    clf = IsolationForest(behaviour='new', max_samples=100,
                          random_state=rng, contamination='auto')
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)

    # plot the line, the samples, and the nearest vectors to the plane
    xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("IsolationForest")
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                     s=20, edgecolor='k')
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                     s=20, edgecolor='k')
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                    s=20, edgecolor='k')
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.legend([b1, b2, c],
               ["training observations",
                "new regular observations", "new abnormal observations"],
               loc="upper left")
    plt.show()


def plot_lof_outlier_detection():
    """
    =================================================
    Outlier detection with Local Outlier Factor (LOF)
    =================================================

    The Local Outlier Factor (LOF) algorithm is an unsupervised anomaly detection
    method which computes the local density deviation of a given data point with
    respect to its neighbors. It considers as outliers the samples that have a
    substantially lower density than their neighbors. This example shows how to
    use LOF for outlier detection which is the default use case of this estimator
    in scikit-learn. Note that when LOF is used for outlier detection it has no
    predict, decision_function and score_samples methods. See
    :ref:`User Guide <outlier_detection>`: for details on the difference between
    outlier detection and novelty detection and how to use LOF for novelty
    detection.

    The number of neighbors considered (parameter n_neighbors) is typically
    set 1) greater than the minimum number of samples a cluster has to contain,
    so that other samples can be local outliers relative to this cluster, and 2)
    smaller than the maximum number of close by samples that can potentially be
    local outliers.
    In practice, such informations are generally not available, and taking
    n_neighbors=20 appears to work well in general.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import LocalOutlierFactor

    print(__doc__)

    np.random.seed(42)

    # Generate train data
    X_inliers = 0.3 * np.random.randn(100, 2)
    X_inliers = np.r_[X_inliers + 2, X_inliers - 2]

    # Generate some outliers
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    X = np.r_[X_inliers, X_outliers]

    n_outliers = len(X_outliers)
    ground_truth = np.ones(len(X), dtype=int)
    ground_truth[-n_outliers:] = -1

    # fit the model for outlier detection (default)
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    # use fit_predict to compute the predicted labels of the training samples
    # (when LOF is used for outlier detection, the estimator has no predict,
    # decision_function and score_samples methods).
    y_pred = clf.fit_predict(X)
    n_errors = (y_pred != ground_truth).sum()
    X_scores = clf.negative_outlier_factor_

    plt.title("Local Outlier Factor (LOF)")
    plt.scatter(X[:, 0], X[:, 1], color='k', s=3., label='Data points')
    # plot circles with radius proportional to the outlier scores
    radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    plt.scatter(X[:, 0], X[:, 1], s=1000 * radius, edgecolors='r',
                facecolors='none', label='Outlier scores')
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.xlabel("prediction errors: %d" % (n_errors))
    legend = plt.legend(loc='upper left')
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]
    plt.show()


if __name__ == '__main__':
    plot_anomaly_comparison()
