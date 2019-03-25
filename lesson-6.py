"""
特征提取
数据预处理
核方法
"""


def example():
    """
    特征提取-将类似文本图片类的数据转化为机器可识别
    """

    # 词袋模型
    # 将文本转化为稀疏矩阵
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    corpus = [
        'This is the first document.',
        'This is the second second document.',
        'And the third one.',
        'Is this the first document?',
        'Is this the first document?',
    ]
    X = vectorizer.fit_transform(corpus)

    analyze = vectorizer.build_analyzer()
    print(analyze("Is this the first document?"))
    print(vectorizer.get_feature_names())
    print(X.toarray())

    print(vectorizer.transform(['Something completely new.']).toarray())

    # 参数 max_df、min_df 出现频率   ngram_range 词语组合    stop_words 停用词
    # 属性 vocabulary_ 词汇表  stop_words_ 停用词
    # 方法 fit_transform 转换 inverse_transform 逆转换  get_feature_names 词袋模型的词库

    # 2 - grams
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                        token_pattern=r'\b\w+\b', min_df=1)
    analyze = bigram_vectorizer.build_analyzer()
    print(analyze('Bi-grams are cool!'))
    X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
    print(X_2)
    print(bigram_vectorizer.get_feature_names())

    """tf - idf 
    一个词语在一篇文章中出现次数越多, 同时在所有文档中出现次数越少, 越能够代表该文章
    """
    # tf - idf
    # TfidfTransformer 转换词袋模型输出  TfidfVectorizer 转换原始数据
    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(X_2)
    print(tfidf.toarray())

    """异常点和新奇点检测
        EllipticEnvelope 假设正态分布，找到部分 样本使得方差最小
        IsolationForest  孤立森林，随意切割，深度越浅越孤立
        LocalOutlierFactor  局部异常因子， 密度相关
        svm.OneClassSVM 单类svm, 超平面包裹数据样例
    """
    plot_anomaly_comparison()

    """数据预处理
    标准化
    最大最小变换
    categorical 类型特征变换
    缺失值处理
    其他可在 from sklearn import preprocessing 中自行阅读
    """
    # 标准化 0，1 正态分布 preprocessing.scale 或 preprocessing.StandardScaler()
    from sklearn import preprocessing
    import numpy as np
    X_train = np.array([[1., -1., 2.],
                        [2., 0., 0.],
                        [0., 1., -1.]])
    X_scaled = preprocessing.scale(X_train)

    print(X_scaled)

    scaler = preprocessing.StandardScaler().fit(X_train)
    print(scaler)
    print(scaler.mean_)
    print(scaler.scale_)
    print(scaler.transform(X_train))

    # 最大最小变换
    X_train = np.array([[1., -1., 2.],
                        [2., 0., 0.],
                        [0., 1., -1.]])

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    print(X_train_minmax)

    # categorical
    # OrdinalEncoder：相同特征占用一列

    enc = preprocessing.OrdinalEncoder()
    X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox'],
         ['fw', 'from Europe', 'uses Firefox']]
    enc.fit(X)

    print(enc.transform([['male', 'from US', 'uses Safari']]))

    # OneHotEncoder：和词袋模型相同
    enc = preprocessing.OneHotEncoder()
    X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
    enc.fit(X)

    print(enc.transform([['female', 'from US', 'uses Safari'],
                         ['male', 'from Europe', 'uses Safari']]).toarray())

    # 缺失值处理
    # strategy ：mean、median、most_frequent、constant
    import numpy as np
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit([[1, 2], [np.nan, 3], [7, 6]])

    X = [[np.nan, 2], [6, np.nan], [7, 6]]
    print(imp.transform(X))


def plot_anomaly_comparison():
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


if __name__ == '__main__':
    pass
