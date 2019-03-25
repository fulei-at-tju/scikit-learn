"""
======================
Decision Trees（决策树）
======================

可以分类（DecisionTreeClassifier）:id3信息增益，c4.5信息增益率，cart基尼系数
也可以回归（DecisionTreeRegressor）:计算最小损失分割点，cart基尼系数

复杂度：
    平衡二叉树O(n_samples * n_features * log(n_samples))
    查询O(log(n_samples))

https://blog.csdn.net/weixin_40604987/article/details/79296427

https://scikit-learn.org/stable/modules/tree.html

Some advantages of decision trees are:
    Simple to understand and to interpret. Trees can be visualised.
    Requires little data preparation. Other techniques often require data normalisation, dummy variables need to be created and blank values to be removed. Note however that this module does not support missing values.
    The cost of using the tree (i.e., predicting data) is logarithmic in the number of data points used to train the tree.
    Able to handle both numerical and categorical data. Other techniques are usually specialised in analysing datasets that have only one type of variable. See algorithms for more information.
    Able to handle multi-output problems.
    Uses a white box model. If a given situation is observable in a model, the explanation for the condition is easily explained by boolean logic. By contrast, in a black box model (e.g., in an artificial neural network), results may be more difficult to interpret.
    Possible to validate a model using statistical tests. That makes it possible to account for the reliability of the model.
    Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.

The disadvantages of decision trees include:
    Decision-tree learners can create over-complex trees that do not generalise the data well. This is called overfitting. Mechanisms such as pruning (not currently supported), setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.
    Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.
    The problem of learning an optimal decision tree is known to be NP-complete under several aspects of optimality and even for simple concepts. Consequently, practical decision-tree learning algorithms are based on heuristic algorithms such as the greedy algorithm where locally optimal decisions are made at each node. Such algorithms cannot guarantee to return the globally optimal decision tree. This can be mitigated by training multiple trees in an ensemble learner, where the features and samples are randomly sampled with replacement.
    There are concepts that are hard to learn because decision trees do not express them easily, such as XOR, parity or multiplexer problems.
    Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.

Tips on practical use:
    过拟合：Decision trees tend to overfit on data with a large number of features. Getting the right ratio of samples to number of features is important, since a tree with few samples in high dimensional space is very likely to overfit.
    降维、特征选择：Consider performing dimensionality reduction (PCA, ICA, or Feature selection) beforehand to give your tree a better chance of finding features that are discriminative.
    可视化及逐步加深层数：Visualise your tree as you are training by using the export function. Use max_depth=3 as an initial tree depth to get a feel for how the tree is fitting to your data, and then increase the depth.
    用max_depth防止过拟合：Remember that the number of samples required to populate the tree doubles for each additional level the tree grows to. Use max_depth to control the size of the tree to prevent overfitting.
    Use min_samples_split or min_samples_leaf to ensure that multiple samples inform every decision in the tree, by controlling which splits will be considered. A very small number will usually mean the tree will overfit, whereas a large number will prevent the tree from learning the data. Try min_samples_leaf=5 as an initial value. If the sample size varies greatly, a float number can be used as percentage in these two parameters. While min_samples_split can create arbitrarily small leaves, min_samples_leaf guarantees that each leaf has a minimum size, avoiding low-variance, over-fit leaf nodes in regression problems. For classification with few classes, min_samples_leaf=1 is often the best choice.
    Balance your dataset before training to prevent the tree from being biased toward the classes that are dominant. Class balancing can be done by sampling an equal number of samples from each class, or preferably by normalizing the sum of the sample weights (sample_weight) for each class to the same value. Also note that weight-based pre-pruning criteria, such as min_weight_fraction_leaf, will then be less biased toward dominant classes than criteria that are not aware of the sample weights, like min_samples_leaf.
    If the samples are weighted, it will be easier to optimize the tree structure using weight-based pre-pruning criterion such as min_weight_fraction_leaf, which ensure that leaf nodes contain at least a fraction of the overall sum of the sample weights.
    All decision trees use np.float32 arrays internally. If training data is not in this format, a copy of the dataset will be made.
    转化稀疏矩阵：If the input matrix X is very sparse, it is recommended to convert to sparse csc_matrix before calling fit and sparse csr_matrix before calling predict. Training time can be orders of magnitude faster for a sparse matrix input compared to a dense matrix when features have zero values in most of the samples.

"""


def graphviz_export():
    from sklearn.datasets import load_iris
    from sklearn import tree
    import graphviz
    iris = load_iris()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(iris.data, iris.target)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree_iris")
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graphviz.Source(dot_data)


def classification():
    from sklearn import tree
    X = [[0, 0], [1, 1]]
    Y = [0, 1]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    print(clf.predict([[2., 2.]]))


def plot_iris():
    """
    ================================================================
    Plot the decision surface of a decision tree on the iris dataset
    ================================================================

    Plot the decision surface of a decision tree trained on pairs
    of features of the iris dataset.

    See :ref:`decision tree <tree>` for more information on the estimator.

    For each pair of iris features, the decision tree learns decision
    boundaries made of combinations of simple thresholding rules inferred from
    the training samples.
    """
    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    # Parameters
    n_classes = 3
    plot_colors = "ryb"
    plot_step = 0.02

    # Load data
    iris = load_iris()

    for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                    [1, 2], [1, 3], [2, 3]]):
        # We only take the two corresponding features
        X = iris.data[:, pair]
        y = iris.target

        # Train
        clf = DecisionTreeClassifier().fit(X, y)

        # Plot the decision boundary
        plt.subplot(2, 3, pairidx + 1)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

        plt.xlabel(iris.feature_names[pair[0]])
        plt.ylabel(iris.feature_names[pair[1]])

        # Plot the training points
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                        cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

    plt.suptitle("Decision surface of a decision tree using paired features")
    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    plt.axis("tight")
    plt.show()


def plot_tree_regression():
    """
    ===================================================================
    Decision Tree Regression
    ===================================================================

    A 1D regression with decision tree.

    The :ref:`decision trees <tree>` is
    used to fit a sine curve with addition noisy observation. As a result, it
    learns local linear regressions approximating the sine curve.

    We can see that if the maximum depth of the tree (controlled by the
    `max_depth` parameter) is set too high, the decision trees learn too fine
    details of the training data and learn from the noise, i.e. they overfit.
    """
    print(__doc__)

    # Import the necessary modules and libraries
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    import matplotlib.pyplot as plt

    # Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(5 * rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(16))

    # Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_1.fit(X, y)
    regr_2.fit(X, y)

    # Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)

    # Plot the results
    plt.figure()
    plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
    plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
    plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()


def plot_tree_regression_multioutput():
    """
    ===================================================================
    Multi-output Decision Tree Regression
    ===================================================================

    An example to illustrate multi-output regression with decision tree.

    The :ref:`decision trees <tree>`
    is used to predict simultaneously the noisy x and y observations of a circle
    given a single underlying feature. As a result, it learns local linear
    regressions approximating the circle.

    We can see that if the maximum depth of the tree (controlled by the
    `max_depth` parameter) is set too high, the decision trees learn too fine
    details of the training data and learn from the noise, i.e. they overfit.
    """
    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeRegressor

    # Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
    y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
    y[::5, :] += (0.5 - rng.rand(20, 2))

    # Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_3 = DecisionTreeRegressor(max_depth=8)
    regr_1.fit(X, y)
    regr_2.fit(X, y)
    regr_3.fit(X, y)

    # Predict
    X_test = np.arange(-100.0, 100.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)
    y_3 = regr_3.predict(X_test)

    # Plot the results
    plt.figure()
    s = 25
    plt.scatter(y[:, 0], y[:, 1], c="navy", s=s, edgecolor="black", label="data")
    plt.scatter(y_1[:, 0], y_1[:, 1], c="cornflowerblue", s=s, edgecolor="black", label="max_depth=2")
    plt.scatter(y_2[:, 0], y_2[:, 1], c="red", s=s, edgecolor="black", label="max_depth=5")
    plt.scatter(y_3[:, 0], y_3[:, 1], c="orange", s=s, edgecolor="black", label="max_depth=8")
    plt.xlim([-6, 6])
    plt.ylim([-6, 6])
    plt.xlabel("target 1")
    plt.ylabel("target 2")
    plt.title("Multi-output Decision Tree Regression")
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    plot_tree_regression()
