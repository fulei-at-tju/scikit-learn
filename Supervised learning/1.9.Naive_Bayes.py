"""
=========================
Naive Bayes ( 朴素贝叶斯 )

['BernoulliNB', 'GaussianNB', 'MultinomialNB', 'ComplementNB']
伯努利，高斯，多项式
=========================

"""


def gaussian_naive_bayes():
    from sklearn import datasets
    iris = datasets.load_iris()
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
    print(y_pred)
    print("Number of mislabeled points out of a total %d points : %d"
          % (iris.data.shape[0], (iris.target != y_pred).sum()))


if __name__ == '__main__':
    gaussian_naive_bayes()
