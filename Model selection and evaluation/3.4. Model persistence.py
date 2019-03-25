"""
==============================
3.4. Model persistence(持久化）
==============================
"""


def example():
    from sklearn import svm
    from sklearn import datasets
    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf.fit(X, y)

    import pickle
    s = pickle.dumps(clf)
    clf2 = pickle.loads(s)
    clf2.predict(X[0:1])

    print(y[0])

    with open('clf.pck', 'wb') as f:
        pickle.dump(clf, f)

    with open('clf.pck', 'rb') as f:
        clf3 = pickle.load(f)
        print(clf3.predict(X[0:1]))

    from joblib import dump, load
    dump(clf, 'clf.joblib')
    clf4 = load('clf.joblib')
    print(clf4.predict(X[0:1]))


if __name__ == '__main__':
    pass
