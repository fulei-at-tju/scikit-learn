"""
===================================
4.3. Preprocessing data(数据预处理）
===================================

标准化:
    0-1高斯分布
        preprocessing.StandardScaler()

    MaxAbsScaler and maxabs_scale :可用于稀疏矩阵
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    robust_scale and RobustScaler :用于离群点

文本分类特征处理：
    preprocessing.OrdinalEncoder（）
    preprocessing.OneHotEncoder（）

离散化：
    K-bins discretization（）
        est = preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2], encode='ordinal').fit(X)
    二值离散化：
        binarizer = preprocessing.Binarizer().fit(X)

"""


def example():
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

    ###########################################
    X_train = np.array([[1., -1., 2.],
                        [2., 0., 0.],
                        [0., 1., -1.]])

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    print(X_train_minmax)

    ############################################
    enc = preprocessing.OrdinalEncoder()
    X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox'],
         ['fw', 'from Europe', 'uses Firefox']]
    enc.fit(X)

    print(enc.transform([['male', 'from US', 'uses Safari']]))

    #############################################
    enc = preprocessing.OneHotEncoder()
    X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
    enc.fit(X)

    print(enc.transform([['female', 'from US', 'uses Safari'],
                         ['male', 'from Europe', 'uses Safari']]).toarray())

    ############################################
    genders = ['female', 'male']
    locations = ['from Africa', 'from Asia', 'from Europe', 'from US']
    browsers = ['uses Chrome', 'uses Firefox', 'uses IE', 'uses Safari']
    enc = preprocessing.OneHotEncoder(categories=[genders, locations, browsers])
    # Note that for there are missing categorical values for the 2nd and 3rd
    # feature
    X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
    enc.fit(X)

    print(enc.transform([['female', 'from Asia', 'uses Chrome']]).toarray())


if __name__ == '__main__':
    pass
