"""
=========================================
4.4. Imputation of missing values(缺失值）
=========================================

impute.SimpleImputer(missing_values=np.nan, strategy='mean')
"""


def example():
    import numpy as np
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit([[1, 2], [np.nan, 3], [7, 6]])

    X = [[np.nan, 2], [6, np.nan], [7, 6]]
    print(imp.transform(X))

    ######################################
    from sklearn.datasets import load_iris
    from sklearn.impute import SimpleImputer, MissingIndicator
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import FeatureUnion, make_pipeline
    from sklearn.tree import DecisionTreeClassifier
    X, y = load_iris(return_X_y=True)
    mask = np.random.randint(0, 2, size=X.shape).astype(np.bool)
    X[mask] = np.nan
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=100,
                                                   random_state=0)

    transformer = FeatureUnion(
        transformer_list=[
            ('features', SimpleImputer(strategy='mean')),
            ('indicators', MissingIndicator())])
    transformer = transformer.fit(X_train, y_train)
    results = transformer.transform(X_test)
    print(results.shape)

    clf = make_pipeline(transformer, DecisionTreeClassifier())
    clf = clf.fit(X_train, y_train)
    results = clf.predict(X_test)
    print(results.shape)


if __name__ == '__main__':
    pass
