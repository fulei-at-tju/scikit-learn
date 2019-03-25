"""
==========================================================
3.2. Tuning the hyper-parameters of an estimator(模型调参)
==========================================================

RandomizedSearchCV 更快
GridSearchCV

优化：
    1.利用好的评价方式
    2.多评价方式一起使用
    3.并行化
    4.设置error_score=0,防止部分模型没有fit方法

    5.模型独有模型选择方法（拟合一个样本和拟合全部样本时间相同）：
        linear_model.ElasticNetCV([l1_ratio, eps, …])	Elastic Net model with iterative fitting along a regularization path.
        linear_model.LarsCV([fit_intercept, …])	Cross-validated Least Angle Regression model.
        linear_model.LassoCV([eps, n_alphas, …])	Lasso linear model with iterative fitting along a regularization path.
        linear_model.LassoLarsCV([fit_intercept, …])	Cross-validated Lasso, using the LARS algorithm.
        linear_model.LogisticRegressionCV([Cs, …])	Logistic Regression CV (aka logit, MaxEnt) classifier.
        linear_model.MultiTaskElasticNetCV([…])	Multi-task L1/L2 ElasticNet with built-in cross-validation.
        linear_model.MultiTaskLassoCV([eps, …])	Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.
        linear_model.OrthogonalMatchingPursuitCV([…])	Cross-validated Orthogonal Matching Pursuit model (OMP).
        linear_model.RidgeCV([alphas, …])	Ridge regression with built-in cross-validation.
        linear_model.RidgeClassifierCV([alphas, …])	Ridge classifier with built-in cross-validation.

    6.集成方法：bagging，抽样的剩余样本
"""


def plot_grid_search_digits():
    """
    ============================================================
    Parameter estimation using grid search with cross-validation
    ============================================================

    This examples shows how a classifier is optimized by cross-validation,
    which is done using the :class:`sklearn.model_selection.GridSearchCV` object
    on a development set that comprises only half of the available labeled data.

    The performance of the selected hyper-parameters and trained model is
    then measured on a dedicated evaluation set that was not used during
    the model selection step.

    More details on tools available for model selection can be found in the
    sections on :ref:`cross_validation` and :ref:`grid_search`.

    """

    from __future__ import print_function

    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.svm import SVC

    print(__doc__)

    # Loading the Digits dataset
    digits = datasets.load_digits()

    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    # Note the problem is too easy: the hyperparameter plateau is too flat and the
    # output model is the same for precision and recall with ties in quality.


def plot_randomized_search():
    """
    =========================================================================
    Comparing randomized search and grid search for hyperparameter estimation
    =========================================================================

    Compare randomized search and grid search for optimizing hyperparameters of a
    random forest.
    All parameters that influence the learning are searched simultaneously
    (except for the number of estimators, which poses a time / quality tradeoff).

    The randomized search and the grid search explore exactly the same space of
    parameters. The result in parameter settings is quite similar, while the run
    time for randomized search is drastically lower.

    The performance is slightly worse for the randomized search, though this
    is most likely a noise effect and would not carry over to a held-out test set.

    Note that in practice, one would not search over this many different parameters
    simultaneously using grid search, but pick only the ones deemed most important.

    RandomizedSearchCV更快
    """
    print(__doc__)

    import numpy as np

    from time import time
    from scipy.stats import randint as sp_randint

    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier

    # get some data
    digits = load_digits()
    X, y = digits.data, digits.target

    # build a classifier
    clf = RandomForestClassifier(n_estimators=20)

    # Utility function to report best scores
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(2, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5)

    start = time()
    random_search.fit(X, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    # use a full grid over all parameters
    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [2, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
    start = time()
    grid_search.fit(X, y)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)


def plot_multi_metric_evaluation():
    """
    ============================================================================
    Demonstration of multi-metric evaluation on cross_val_score and GridSearchCV
    ============================================================================

    Multiple metric parameter search can be done by setting the ``scoring``
    parameter to a list of metric scorer names or a dict mapping the scorer names
    to the scorer callables.

    The scores of all the scorers are available in the ``cv_results_`` dict at keys
    ending in ``'_<scorer_name>'`` (``'mean_test_precision'``,
    ``'rank_test_precision'``, etc...)

    The ``best_estimator_``, ``best_index_``, ``best_score_`` and ``best_params_``
    correspond to the scorer (key) that is set to the ``refit`` attribute.
    """

    # Author: Raghav RV <rvraghav93@gmail.com>
    # License: BSD

    import numpy as np
    from matplotlib import pyplot as plt

    from sklearn.datasets import make_hastie_10_2
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier

    print(__doc__)

    ###############################################################################
    # Running ``GridSearchCV`` using multiple evaluation metrics
    # ----------------------------------------------------------
    #

    X, y = make_hastie_10_2(n_samples=8000, random_state=42)

    # The scorers can be either be one of the predefined metric strings or a scorer
    # callable, like the one returned by make_scorer
    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

    # Setting refit='AUC', refits an estimator on the whole dataset with the
    # parameter setting that has the best cross-validated AUC score.
    # That estimator is made available at ``gs.best_estimator_`` along with
    # parameters like ``gs.best_score_``, ``gs.best_params_`` and
    # ``gs.best_index_``
    gs = GridSearchCV(DecisionTreeClassifier(random_state=42),
                      param_grid={'min_samples_split': range(2, 403, 10)},
                      scoring=scoring, cv=5, refit='AUC', return_train_score=True)
    gs.fit(X, y)
    results = gs.cv_results_

    ###############################################################################
    # Plotting the result
    # -------------------

    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
              fontsize=16)

    plt.xlabel("min_samples_split")
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(0, 402)
    ax.set_ylim(0.73, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_min_samples_split'].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid('off')
    plt.show()


if __name__ == '__main__':
    pass
