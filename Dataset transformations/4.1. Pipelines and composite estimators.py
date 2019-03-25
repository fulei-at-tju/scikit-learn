"""
========================================================
4.1. Pipelines and composite estimators(管道和复合分类器）
========================================================

分类：
    网格搜索-》（特征选择、降维、分类器）

    pipe = Pipeline([
        ('reduce_dim', None),
        ('classify', LinearSVC())
    ])

    N_FEATURES_OPTIONS = [2, 4, 8]
    C_OPTIONS = [1, 10, 100, 1000]
    param_grid = [
        {
            'reduce_dim': [PCA(iterated_power=7), NMF()],
            'reduce_dim__n_components': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },
        {
            'reduce_dim': [SelectKBest(chi2)],
            'reduce_dim__k': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS
        },
    ]
    reducer_labels = ['PCA', 'NMF', 'KBest(chi2)']
    grid = GridSearchCV(pipe, cv=5, n_jobs=1, param_grid=param_grid)

回归：
    改变y的分布进行预测，再变为原始分布

    regr_trans = TransformedTargetRegressor(
        regressor=RidgeCV(),
        transformer=QuantileTransformer(output_distribution='normal'))

    regr_trans = TransformedTargetRegressor(regressor=RidgeCV(),
                                            func=np.log1p,
                                            inverse_func=np.expm1)

特征融合：
    combined_features = FeatureUnion([("pca", PCA(n_components=2)), ("univ_select", SelectKBest(k=1))])

异构数据特征处理：
    column_trans = ColumnTransformer(
        [('city_category', CountVectorizer(analyzer=lambda x: [x]), 'city'),
         ('title_bow', CountVectorizer(), 'title')],
        remainder='drop')

    column_trans.fit(X)
"""


def examples():
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    pipe = Pipeline(estimators)
    print(pipe)
    print(pipe.steps[0])
    print(pipe.named_steps['reduce_dim'])

    pipe.set_params(clf__C=10)
    print(pipe.named_steps['clf'])

    ###################################################
    # 网格搜索，搜索管道中的参数(重要)
    from sklearn.model_selection import GridSearchCV
    param_grid = dict(reduce_dim__n_components=[2, 5, 10],
                      clf__C=[0.1, 10, 100])
    grid_search = GridSearchCV(pipe, param_grid=param_grid)
    print(grid_search)

    ###################################################
    # 网格搜索，搜索管道中的参数(重要)
    from sklearn.linear_model import LogisticRegression

    param_grid = dict(reduce_dim=[None, PCA(5), PCA(10)],
                      clf=[SVC(), LogisticRegression()],
                      clf__C=[0.1, 10, 100])  # 多个可组成列表
    grid_search = GridSearchCV(pipe, param_grid=param_grid)
    print(grid_search)

    ###################################################
    from sklearn.pipeline import make_pipeline
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import Binarizer
    pipe = make_pipeline(Binarizer(), MultinomialNB())
    print(pipe)

    ###################################################
    # 利用memory减少重复计算
    from tempfile import mkdtemp
    from shutil import rmtree
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    cachedir = mkdtemp()
    pipe = Pipeline(estimators, memory=cachedir)
    print(pipe)

    # Clear the cache directory when you don't need it anymore
    rmtree(cachedir)

    #####################################################
    # 　Transforming target in regression
    import numpy as np
    from sklearn.datasets import load_boston
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    boston = load_boston()
    X = boston.data
    y = boston.target
    transformer = QuantileTransformer(output_distribution='normal')
    regressor = LinearRegression()
    regr = TransformedTargetRegressor(regressor=regressor,
                                      transformer=transformer)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    regr.fit(X_train, y_train)

    print('R2 score: {0:.2f}'.format(regr.score(X_test, y_test)))

    raw_target_regr = LinearRegression().fit(X_train, y_train)
    print('R2 score: {0:.2f}'.format(raw_target_regr.score(X_test, y_test)))

    ##########################################################
    # 对每列数据进行处理-预处理
    import pandas as pd
    X = pd.DataFrame(
        {'city': ['London', 'London', 'Paris', 'Sallisaw'],
         'title': ["His Last Bow", "How Watson Learned the Trick",
                   "A Moveable Feast", "The Grapes of Wrath"],
         'expert_rating': [5, 3, 4, 5],
         'user_rating': [4, 5, 4, 3]})

    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    column_trans = ColumnTransformer(
        [('city_category', CountVectorizer(analyzer=lambda x: [x]), 'city'),
         ('title_bow', CountVectorizer(), 'title')],
        remainder='drop')

    print(column_trans.fit(X))
    print(column_trans.get_feature_names())
    print(column_trans.transform(X).toarray())


def plot_feature_selection_pipeline():
    """
    ==================
    Pipeline Anova SVM
    ==================

    Simple usage of Pipeline that runs successively a univariate
    feature selection with anova and then a C-SVM of the selected features.
    """
    from sklearn import svm
    from sklearn.datasets import samples_generator
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    print(__doc__)

    # import some data to play with
    X, y = samples_generator.make_classification(
        n_features=20, n_informative=3, n_redundant=0, n_classes=4,
        n_clusters_per_class=2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # ANOVA SVM-C
    # 1) anova filter, take 3 best ranked features
    anova_filter = SelectKBest(f_regression, k=3)
    # 2) svm
    clf = svm.SVC(kernel='rbf')

    anova_svm = make_pipeline(anova_filter, clf)
    anova_svm.fit(X_train, y_train)
    y_pred = anova_svm.predict(X_test)
    print(classification_report(y_test, y_pred))


def plot_digits_pipe():
    # !/usr/bin/python
    # -*- coding: utf-8 -*-

    """
    =========================================================
    Pipelining: chaining a PCA and a logistic regression
    =========================================================

    The PCA does an unsupervised dimensionality reduction, while the logistic
    regression does the prediction.

    We use a GridSearchCV to set the dimensionality of the PCA

    """
    print(__doc__)

    # Code source: Gaël Varoquaux
    # Modified for documentation by Jaques Grobler
    # License: BSD 3 clause

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    from sklearn import datasets
    from sklearn.decomposition import PCA
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV

    # Define a pipeline to search for the best combination of PCA truncation
    # and classifier regularization.
    logistic = SGDClassifier(loss='log', penalty='l2', early_stopping=True,
                             max_iter=10000, tol=1e-5, random_state=0)
    pca = PCA()
    pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

    digits = datasets.load_digits()
    X_digits = digits.data
    y_digits = digits.target

    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {
        'pca__n_components': [5, 20, 30, 40, 50, 64],
        'logistic__alpha': np.logspace(-4, 4, 5),
    }
    search = GridSearchCV(pipe, param_grid, iid=False, cv=5,
                          return_train_score=False)
    search.fit(X_digits, y_digits)
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    # Plot the PCA spectrum
    pca.fit(X_digits)

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
    ax0.plot(pca.explained_variance_ratio_, linewidth=2)
    ax0.set_ylabel('PCA explained variance')

    ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='n_components chosen')
    ax0.legend(prop=dict(size=12))

    # For each number of components, find the best classifier results
    results = pd.DataFrame(search.cv_results_)
    components_col = 'param_pca__n_components'
    best_clfs = results.groupby(components_col).apply(
        lambda g: g.nlargest(1, 'mean_test_score'))

    best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
                   legend=False, ax=ax1)
    ax1.set_ylabel('Classification accuracy (val)')
    ax1.set_xlabel('n_components')

    plt.tight_layout()
    plt.show()


def plot_transformed_target():
    # !/usr/bin/env python
    # -*- coding: utf-8 -*-

    """
    ======================================================
    Effect of transforming the targets in regression model
    ======================================================

    In this example, we give an overview of the
    :class:`sklearn.compose.TransformedTargetRegressor`. Two examples
    illustrate the benefit of transforming the targets before learning a linear
    regression model. The first example uses synthetic data while the second
    example is based on the Boston housing data set.

    """

    # Author: Guillaume Lemaitre <guillaume.lemaitre@inria.fr>
    # License: BSD 3 clause

    from __future__ import print_function, division

    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from distutils.version import LooseVersion

    print(__doc__)

    ###############################################################################
    # Synthetic example
    ###############################################################################

    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import RidgeCV
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.metrics import median_absolute_error, r2_score

    # `normed` is being deprecated in favor of `density` in histograms
    if LooseVersion(matplotlib.__version__) >= '2.1':
        density_param = {'density': True}
    else:
        density_param = {'normed': True}

    ###############################################################################
    # A synthetic random regression problem is generated. The targets ``y`` are
    # modified by: (i) translating all targets such that all entries are
    # non-negative and (ii) applying an exponential function to obtain non-linear
    # targets which cannot be fitted using a simple linear model.
    #
    # Therefore, a logarithmic (`np.log1p`) and an exponential function
    # (`np.expm1`) will be used to transform the targets before training a linear
    # regression model and using it for prediction.

    X, y = make_regression(n_samples=10000, noise=100, random_state=0)
    y = np.exp((y + abs(y.min())) / 200)
    y_trans = np.log1p(y)

    ###############################################################################
    # The following illustrate the probability density functions of the target
    # before and after applying the logarithmic functions.

    f, (ax0, ax1) = plt.subplots(1, 2)

    ax0.hist(y, bins=100, **density_param)
    ax0.set_xlim([0, 2000])
    ax0.set_ylabel('Probability')
    ax0.set_xlabel('Target')
    ax0.set_title('Target distribution')

    ax1.hist(y_trans, bins=100, **density_param)
    ax1.set_ylabel('Probability')
    ax1.set_xlabel('Target')
    ax1.set_title('Transformed target distribution')

    f.suptitle("Synthetic data", y=0.035)
    f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    ###############################################################################
    # At first, a linear model will be applied on the original targets. Due to the
    # non-linearity, the model trained will not be precise during the
    # prediction. Subsequently, a logarithmic function is used to linearize the
    # targets, allowing better prediction even with a similar linear model as
    # reported by the median absolute error (MAE).

    f, (ax0, ax1) = plt.subplots(1, 2, sharey=True)

    regr = RidgeCV()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    ax0.scatter(y_test, y_pred)
    ax0.plot([0, 2000], [0, 2000], '--k')
    ax0.set_ylabel('Target predicted')
    ax0.set_xlabel('True Target')
    ax0.set_title('Ridge regression \n without target transformation')
    ax0.text(100, 1750, r'$R^2$=%.2f, MAE=%.2f' % (
        r2_score(y_test, y_pred), median_absolute_error(y_test, y_pred)))
    ax0.set_xlim([0, 2000])
    ax0.set_ylim([0, 2000])

    regr_trans = TransformedTargetRegressor(regressor=RidgeCV(),
                                            func=np.log1p,
                                            inverse_func=np.expm1)
    regr_trans.fit(X_train, y_train)
    y_pred = regr_trans.predict(X_test)

    ax1.scatter(y_test, y_pred)
    ax1.plot([0, 2000], [0, 2000], '--k')
    ax1.set_ylabel('Target predicted')
    ax1.set_xlabel('True Target')
    ax1.set_title('Ridge regression \n with target transformation')
    ax1.text(100, 1750, r'$R^2$=%.2f, MAE=%.2f' % (
        r2_score(y_test, y_pred), median_absolute_error(y_test, y_pred)))
    ax1.set_xlim([0, 2000])
    ax1.set_ylim([0, 2000])

    f.suptitle("Synthetic data", y=0.035)
    f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    ###############################################################################
    # Real-world data set
    ###############################################################################

    ###############################################################################
    # In a similar manner, the boston housing data set is used to show the impact
    # of transforming the targets before learning a model. In this example, the
    # targets to be predicted corresponds to the weighted distances to the five
    # Boston employment centers.

    from sklearn.datasets import load_boston
    from sklearn.preprocessing import QuantileTransformer, quantile_transform

    dataset = load_boston()
    target = np.array(dataset.feature_names) == "DIS"
    X = dataset.data[:, np.logical_not(target)]
    y = dataset.data[:, target].squeeze()
    y_trans = quantile_transform(dataset.data[:, target],
                                 output_distribution='normal').squeeze()

    ###############################################################################
    # A :class:`sklearn.preprocessing.QuantileTransformer` is used such that the
    # targets follows a normal distribution before applying a
    # :class:`sklearn.linear_model.RidgeCV` model.

    f, (ax0, ax1) = plt.subplots(1, 2)

    ax0.hist(y, bins=100, **density_param)
    ax0.set_ylabel('Probability')
    ax0.set_xlabel('Target')
    ax0.set_title('Target distribution')

    ax1.hist(y_trans, bins=100, **density_param)
    ax1.set_ylabel('Probability')
    ax1.set_xlabel('Target')
    ax1.set_title('Transformed target distribution')

    f.suptitle("Boston housing data: distance to employment centers", y=0.035)
    f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    ###############################################################################
    # The effect of the transformer is weaker than on the synthetic data. However,
    # the transform induces a decrease of the MAE.

    f, (ax0, ax1) = plt.subplots(1, 2, sharey=True)

    regr = RidgeCV()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)

    ax0.scatter(y_test, y_pred)
    ax0.plot([0, 10], [0, 10], '--k')
    ax0.set_ylabel('Target predicted')
    ax0.set_xlabel('True Target')
    ax0.set_title('Ridge regression \n without target transformation')
    ax0.text(1, 9, r'$R^2$=%.2f, MAE=%.2f' % (
        r2_score(y_test, y_pred), median_absolute_error(y_test, y_pred)))
    ax0.set_xlim([0, 10])
    ax0.set_ylim([0, 10])

    regr_trans = TransformedTargetRegressor(
        regressor=RidgeCV(),
        transformer=QuantileTransformer(output_distribution='normal'))

    regr_trans.fit(X_train, y_train)
    y_pred = regr_trans.predict(X_test)

    ax1.scatter(y_test, y_pred)
    ax1.plot([0, 10], [0, 10], '--k')
    ax1.set_ylabel('Target predicted')
    ax1.set_xlabel('True Target')
    ax1.set_title('Ridge regression \n with target transformation')
    ax1.text(1, 9, r'$R^2$=%.2f, MAE=%.2f' % (
        r2_score(y_test, y_pred), median_absolute_error(y_test, y_pred)))
    ax1.set_xlim([0, 10])
    ax1.set_ylim([0, 10])

    f.suptitle("Boston housing data: distance to employment centers", y=0.035)
    f.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

    plt.show()


def plot_feature_union():
    """
    =================================================
    Concatenating multiple feature extraction methods
    =================================================

    In many real-world examples, there are many ways to extract features from a
    dataset. Often it is beneficial to combine several methods to obtain good
    performance. This example shows how to use ``FeatureUnion`` to combine
    features obtained by PCA and univariate selection.

    Combining features using this transformer has the benefit that it allows
    cross validation and grid searches over the whole process.

    The combination used in this example is not particularly helpful on this
    dataset and is only used to illustrate the usage of FeatureUnion.
    """

    # Author: Andreas Mueller <amueller@ais.uni-bonn.de>
    #
    # License: BSD 3 clause

    from __future__ import print_function
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest

    iris = load_iris()

    X, y = iris.data, iris.target

    # This dataset is way too high-dimensional. Better do PCA:
    pca = PCA(n_components=2)

    # Maybe some original features where good, too?
    selection = SelectKBest(k=1)

    # Build estimator from PCA and Univariate selection:

    combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])

    # Use combined features to transform dataset:
    X_features = combined_features.fit(X, y).transform(X)
    print("Combined space has", X_features.shape[1], "features")

    svm = SVC(kernel="linear")

    # Do grid search over k, n_components and C:

    pipeline = Pipeline([("features", combined_features), ("svm", svm)])

    param_grid = dict(features__pca__n_components=[1, 2, 3],
                      features__univ_select__k=[1, 2],
                      svm__C=[0.1, 1, 10])

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=10)
    grid_search.fit(X, y)
    print(grid_search.best_estimator_)


def plot_column_transformer():
    """
    ==================================================
    Column Transformer with Heterogeneous Data Sources
    ==================================================

    Datasets can often contain components of that require different feature
    extraction and processing pipelines.  This scenario might occur when:

    1. Your dataset consists of heterogeneous data types (e.g. raster images and
       text captions)
    2. Your dataset is stored in a Pandas DataFrame and different columns
       require different processing pipelines.

    This example demonstrates how to use
    :class:`sklearn.compose.ColumnTransformer` on a dataset containing
    different types of features.  We use the 20-newsgroups dataset and compute
    standard bag-of-words features for the subject line and body in separate
    pipelines as well as ad hoc features on the body. We combine them (with
    weights) using a ColumnTransformer and finally train a classifier on the
    combined set of features.

    The choice of features is not particularly helpful, but serves to illustrate
    the technique.
    """

    # Author: Matt Terry <matt.terry@gmail.com>
    #
    # License: BSD 3 clause
    from __future__ import print_function

    import numpy as np

    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.datasets.twenty_newsgroups import strip_newsgroup_footer
    from sklearn.datasets.twenty_newsgroups import strip_newsgroup_quoting
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.svm import LinearSVC

    class TextStats(BaseEstimator, TransformerMixin):
        """Extract features from each document for DictVectorizer"""

        def fit(self, x, y=None):
            return self

        def transform(self, posts):
            return [{'length': len(text),
                     'num_sentences': text.count('.')}
                    for text in posts]

    class SubjectBodyExtractor(BaseEstimator, TransformerMixin):
        """Extract the subject & body from a usenet post in a single pass.

        Takes a sequence of strings and produces a dict of sequences.  Keys are
        `subject` and `body`.
        """

        def fit(self, x, y=None):
            return self

        def transform(self, posts):
            # construct object dtype array with two columns
            # first column = 'subject' and second column = 'body'
            features = np.empty(shape=(len(posts), 2), dtype=object)
            for i, text in enumerate(posts):
                headers, _, bod = text.partition('\n\n')
                bod = strip_newsgroup_footer(bod)
                bod = strip_newsgroup_quoting(bod)
                features[i, 1] = bod

                prefix = 'Subject:'
                sub = ''
                for line in headers.split('\n'):
                    if line.startswith(prefix):
                        sub = line[len(prefix):]
                        break
                features[i, 0] = sub

            return features

    pipeline = Pipeline([
        # Extract the subject & body
        ('subjectbody', SubjectBodyExtractor()),

        # Use ColumnTransformer to combine the features from subject and body
        ('union', ColumnTransformer(
            [
                # Pulling features from the post's subject line (first column)
                ('subject', TfidfVectorizer(min_df=50), 0),

                # Pipeline for standard bag-of-words model for body (second column)
                ('body_bow', Pipeline([
                    ('tfidf', TfidfVectorizer()),
                    ('best', TruncatedSVD(n_components=50)),
                ]), 1),

                # Pipeline for pulling ad hoc features from post's body
                ('body_stats', Pipeline([
                    ('stats', TextStats()),  # returns a list of dicts
                    ('vect', DictVectorizer()),  # list of dicts -> feature matrix
                ]), 1),
            ],

            # weight components in ColumnTransformer
            transformer_weights={
                'subject': 0.8,
                'body_bow': 0.5,
                'body_stats': 1.0,
            }
        )),

        # Use a SVC classifier on the combined features
        ('svc', LinearSVC()),
    ])

    # limit the list of categories to make running this example faster.
    categories = ['alt.atheism', 'talk.religion.misc']
    train = fetch_20newsgroups(random_state=1,
                               subset='train',
                               categories=categories,
                               )
    test = fetch_20newsgroups(random_state=1,
                              subset='test',
                              categories=categories,
                              )

    pipeline.fit(train.data, train.target)
    y = pipeline.predict(test.data)
    print(classification_report(y, test.target))


def plot_column_transformer_mixed_types():
    """
    ===================================
    Column Transformer with Mixed Types
    ===================================

    This example illustrates how to apply different preprocessing and
    feature extraction pipelines to different subsets of features,
    using :class:`sklearn.compose.ColumnTransformer`.
    This is particularly handy for the case of datasets that contain
    heterogeneous data types, since we may want to scale the
    numeric features and one-hot encode the categorical ones.

    In this example, the numeric data is standard-scaled after
    mean-imputation, while the categorical data is one-hot
    encoded after imputing missing values with a new category
    (``'missing'``).

    Finally, the preprocessing pipeline is integrated in a
    full prediction pipeline using :class:`sklearn.pipeline.Pipeline`,
    together with a simple classification model.
    """

    # Author: Pedro Morales <part.morales@gmail.com>
    #
    # License: BSD 3 clause

    from __future__ import print_function

    import pandas as pd
    import numpy as np

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, GridSearchCV

    np.random.seed(0)

    # Read data from Titanic dataset.
    titanic_url = ('https://raw.githubusercontent.com/amueller/'
                   'scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv')
    data = pd.read_csv(titanic_url)

    # We will train our classifier with the following features:
    # Numeric Features:
    # - age: float.
    # - fare: float.
    # Categorical Features:
    # - embarked: categories encoded as strings {'C', 'S', 'Q'}.
    # - sex: categories encoded as strings {'female', 'male'}.
    # - pclass: ordinal integers {1, 2, 3}.

    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_features = ['age', 'fare']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = ['embarked', 'sex', 'pclass']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression(solver='lbfgs'))])

    X = data.drop('survived', axis=1)
    y = data['survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf.fit(X_train, y_train)
    print("model score: %.3f" % clf.score(X_test, y_test))

    ###############################################################################
    # Using the prediction pipeline in a grid search
    ###############################################################################
    # Grid search can also be performed on the different preprocessing steps
    # defined in the ``ColumnTransformer`` object, together with the classifier's
    # hyperparameters as part of the ``Pipeline``.
    # We will search for both the imputer strategy of the numeric preprocessing
    # and the regularization parameter of the logistic regression using
    # :class:`sklearn.model_selection.GridSearchCV`.

    param_grid = {
        'preprocessor__num__imputer__strategy': ['mean', 'median'],
        'classifier__C': [0.1, 1.0, 10, 100],
    }

    grid_search = GridSearchCV(clf, param_grid, cv=10, iid=False)
    grid_search.fit(X_train, y_train)

    print(("best logistic regression from grid search: %.3f"
           % grid_search.score(X_test, y_test)))


if __name__ == '__main__':
    # 全量样例(重要)
    plot_column_transformer_mixed_types()
