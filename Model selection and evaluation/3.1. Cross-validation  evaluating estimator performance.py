"""
=================================================================
3.1. Cross-validation: evaluating estimator performance（交叉验证）
=================================================================

('BaseCrossValidator',
'GridSearchCV',
'TimeSeriesSplit',
'KFold',
'GroupKFold',
'GroupShuffleSplit',
'LeaveOneGroupOut',
'LeaveOneOut',
'LeavePGroupsOut',
'LeavePOut',
'RepeatedKFold',
'RepeatedStratifiedKFold',
'ParameterGrid',
'ParameterSampler',
'PredefinedSplit',
'RandomizedSearchCV',
'ShuffleSplit',
'StratifiedKFold',
'StratifiedShuffleSplit',
'check_cv',
'cross_val_predict',
'cross_val_score',
'cross_validate',
'fit_grid_point',
'learning_curve',
'permutation_test_score',
'train_test_split',
'validation_curve')

多评价函数： GridSearchCV, RandomizedSearchCV and cross_validate.
"""


def print_train_test_split():
    """测试 数据集切割"""
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from sklearn import svm

    iris = datasets.load_iris()
    print(iris.data.shape, iris.target.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.4, random_state=0)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    print(clf.score(X_test, y_test))

    ################################
    from sklearn.model_selection import cross_val_score

    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_macro')
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    ################################
    from sklearn.model_selection import ShuffleSplit
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    print(cross_val_score(clf, iris.data, iris.target, cv=cv))

    ################################
    from sklearn import preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.4, random_state=0)
    scaler = preprocessing.StandardScaler().fit(X_train)
    print(scaler.mean_)
    print(scaler.var_)
    X_train_transformed = scaler.transform(X_train)
    clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
    X_test_transformed = scaler.transform(X_test)
    print(clf.score(X_test_transformed, y_test))

    ##################################
    from sklearn.pipeline import make_pipeline
    clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
    print(cross_val_score(clf, iris.data, iris.target, cv=cv))

    ##################################
    # 多个评分函数，字典形式返回
    # cross_validate()
    from sklearn.model_selection import cross_validate
    scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    clf = svm.SVC(kernel='linear', C=1, random_state=0)
    scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,
                            cv=5, return_train_score=False)
    print(scores.keys())

    print(scores['test_recall_macro'])


def plot_roc_crossval():
    """
    =============================================================
    Receiver Operating Characteristic (ROC) with cross validation
    =============================================================

    Example of Receiver Operating Characteristic (ROC) metric to evaluate
    classifier output quality using cross-validation.

    ROC curves typically feature true positive rate on the Y axis, and false
    positive rate on the X axis. This means that the top left corner of the plot is
    the "ideal" point - a false positive rate of zero, and a true positive rate of
    one. This is not very realistic, but it does mean that a larger area under the
    curve (AUC) is usually better.

    The "steepness" of ROC curves is also important, since it is ideal to maximize
    the true positive rate while minimizing the false positive rate.

    This example shows the ROC response of different datasets, created from K-fold
    cross-validation. Taking all of these curves, it is possible to calculate the
    mean area under curve, and see the variance of the curve when the
    training set is split into different subsets. This roughly shows how the
    classifier output is affected by changes in the training data, and how
    different the splits generated by K-fold cross-validation are from one another.

    .. note::

        See also :func:`sklearn.metrics.roc_auc_score`,
                 :func:`sklearn.model_selection.cross_val_score`,
                 :ref:`sphx_glr_auto_examples_model_selection_plot_roc.py`,

    """
    print(__doc__)

    import numpy as np
    from scipy import interp
    import matplotlib.pyplot as plt

    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold

    # #############################################################################
    # Data IO and generation

    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X, y = X[y != 2], y[y != 2]
    n_samples, n_features = X.shape

    # Add noisy features
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # #############################################################################
    # Classification and ROC analysis

    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=6)
    classifier = svm.SVC(kernel='linear', probability=True,
                         random_state=random_state)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


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


def grid_search_text_feature_extraction():
    """
    ==========================================================
    Sample pipeline for text feature extraction and evaluation
    ==========================================================

    The dataset used in this example is the 20 newsgroups dataset which will be
    automatically downloaded and then cached and reused for the document
    classification example.

    You can adjust the number of categories by giving their names to the dataset
    loader or setting them to None to get the 20 of them.

    Here is a sample output of a run on a quad-core machine::

      Loading 20 newsgroups dataset for categories:
      ['alt.atheism', 'talk.religion.misc']
      1427 documents
      2 categories

      Performing grid search...
      pipeline: ['vect', 'tfidf', 'clf']
      parameters:
      {'clf__alpha': (1.0000000000000001e-05, 9.9999999999999995e-07),
       'clf__max_iter': (10, 50, 80),
       'clf__penalty': ('l2', 'elasticnet'),
       'tfidf__use_idf': (True, False),
       'vect__max_n': (1, 2),
       'vect__max_df': (0.5, 0.75, 1.0),
       'vect__max_features': (None, 5000, 10000, 50000)}
      done in 1737.030s

      Best score: 0.940
      Best parameters set:
          clf__alpha: 9.9999999999999995e-07
          clf__max_iter: 50
          clf__penalty: 'elasticnet'
          tfidf__use_idf: True
          vect__max_n: 2
          vect__max_df: 0.75
          vect__max_features: 50000

    """

    # Author: Olivier Grisel <olivier.grisel@ensta.org>
    #         Peter Prettenhofer <peter.prettenhofer@gmail.com>
    #         Mathieu Blondel <mathieu@mblondel.org>
    # License: BSD 3 clause

    from __future__ import print_function

    from pprint import pprint
    from time import time
    import logging

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    print(__doc__)

    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # #############################################################################
    # Load some categories from the training set
    categories = [
        'alt.atheism',
        'talk.religion.misc',
    ]
    # Uncomment the following to do the analysis on all the categories
    # categories = None

    print("Loading 20 newsgroups dataset for categories:")
    print(categories)

    data = fetch_20newsgroups(subset='train', categories=categories)
    print("%d documents" % len(data.filenames))
    print("%d categories" % len(data.target_names))
    print()

    # #############################################################################
    # Define a pipeline combining a text feature extractor with a simple
    # classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])

    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__max_iter': (5,),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        # 'clf__max_iter': (10, 50, 80),
    }

    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, cv=5,
                               n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(data.data, data.target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


if __name__ == '__main__':
    pass
