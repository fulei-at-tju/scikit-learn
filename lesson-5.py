"""
模型评估
交叉验证
性能度量
超参数：GridSearchCV and Pipeline
持久化
"""


def example():
    """数据集切分"""
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from sklearn import svm

    iris = datasets.load_iris()
    print(iris.data.shape, iris.target.shape)  # 150条

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.4, random_state=0)

    print(X_train.shape, y_train.shape)  # 90条

    print(X_test.shape, y_test.shape)  # 60条

    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    clf.score(X_test, y_test)

    """交叉验证"""

    # 1、五折交叉验证
    from sklearn.model_selection import cross_val_score
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    print(scores)

    # 2、修改评分
    scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_macro')
    print(scores)

    # 3、修改cv方法
    # K-fold、Repeated K-Fold、Leave One Out (LOO)、Leave P Out (LPO)、ShuffleSplit、Stratified k-fold
    # Stratified Shuffle Split、Group k-fold
    # https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators
    from sklearn.model_selection import ShuffleSplit
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    scores = cross_val_score(clf, iris.data, iris.target, cv=cv)
    print(scores)

    # 4、多评价指标
    from sklearn.model_selection import cross_validate
    scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    clf = svm.SVC(kernel='linear', C=1, random_state=0)
    scores = cross_validate(clf, iris.data, iris.target, scoring=scoring, cv=5, return_train_score=False)
    print(scores.keys())
    print(scores['test_recall_macro'])

    """性能度量"""
    """
    Classification
        ‘accuracy’	metrics.accuracy_score
        ‘balanced_accuracy’	metrics.balanced_accuracy_score	for binary targets
        ‘average_precision’	metrics.average_precision_score
        ‘brier_score_loss’	metrics.brier_score_loss
        ‘f1’	metrics.f1_score	for binary targets
        ‘f1_micro’	metrics.f1_score	micro-averaged
        ‘f1_macro’	metrics.f1_score	macro-averaged
        ‘f1_weighted’	metrics.f1_score	weighted average
        ‘f1_samples’	metrics.f1_score	by multilabel sample
        ‘neg_log_loss’	metrics.log_loss	requires predict_proba support
        ‘precision’ etc.	metrics.precision_score	suffixes apply as with ‘f1’
        ‘recall’ etc.	metrics.recall_score	suffixes apply as with ‘f1’
        ‘roc_auc’	metrics.roc_auc_score
    Clustering
        ‘adjusted_mutual_info_score’	metrics.adjusted_mutual_info_score
        ‘adjusted_rand_score’	metrics.adjusted_rand_score
        ‘completeness_score’	metrics.completeness_score
        ‘fowlkes_mallows_score’	metrics.fowlkes_mallows_score
        ‘homogeneity_score’	metrics.homogeneity_score
        ‘mutual_info_score’	metrics.mutual_info_score
        ‘normalized_mutual_info_score’	metrics.normalized_mutual_info_score
        ‘v_measure_score’	metrics.v_measure_score
    Regression
        ‘explained_variance’	metrics.explained_variance_score
        ‘neg_mean_absolute_error’	metrics.mean_absolute_error
        ‘neg_mean_squared_error’	metrics.mean_squared_error
        ‘neg_mean_squared_log_error’	metrics.mean_squared_log_error
        ‘neg_median_absolute_error’	metrics.median_absolute_error
        ‘r2’	metrics.r2_score
    """

    # 1、make_score
    from sklearn.svm import LinearSVC
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import confusion_matrix
    X, y = datasets.make_classification(n_classes=2, random_state=0)
    svm = LinearSVC(random_state=0)

    def tn(_y_true, _y_pred): return confusion_matrix(_y_true, _y_pred)[0, 0]

    def fp(_y_true, _y_pred): return confusion_matrix(_y_true, _y_pred)[0, 1]

    def fn(_y_true, _y_pred): return confusion_matrix(_y_true, _y_pred)[1, 0]

    def tp(_y_true, _y_pred): return confusion_matrix(_y_true, _y_pred)[1, 1]

    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
               'fp': make_scorer(fp), 'fn': make_scorer(fn)}
    cv_results = cross_validate(svm.fit(X, y), X, y, scoring=scoring, cv=5)
    print(cv_results['test_tp'])
    print(cv_results['test_fn'])

    # 2、混淆矩阵
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    print(confusion_matrix(y_true, y_pred))

    # 3、r2 1-(y-y预测)^2/(y-y平均）^2
    from sklearn.metrics import r2_score
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    print(r2_score(y_true, y_pred))

    y_true = [[0.5, 1], [-1, 1], [7, -6]]
    y_pred = [[0, 2], [-1, 2], [8, -5]]
    print(r2_score(y_true, y_pred, multioutput='variance_weighted'))

    y_true = [[0.5, 1], [-1, 1], [7, -6]]
    y_pred = [[0, 2], [-1, 2], [8, -5]]
    print(r2_score(y_true, y_pred, multioutput='uniform_average'))
    print(r2_score(y_true, y_pred, multioutput='raw_values'))
    print(r2_score(y_true, y_pred, multioutput=[0.3, 0.7]))

    # 4、roc曲线及宏平均微平均
    plot_roc()

    """超参数  Grid Search 模型选择  Pipelines"""
    # 1、Grid Search
    plot_grid_search_digits()

    # 2、Pipelines
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

    # 2.1增加网格搜索
    from sklearn.model_selection import GridSearchCV
    param_grid = dict(reduce_dim__n_components=[2, 5, 10],
                      clf__C=[0.1, 10, 100])
    grid_search = GridSearchCV(pipe, param_grid=param_grid)
    print(grid_search)

    # 2.1增加网格搜索
    from sklearn.linear_model import LogisticRegression

    param_grid = dict(reduce_dim=[None, PCA(5), PCA(10)],
                      clf=[SVC(), LogisticRegression()],
                      clf__C=[0.1, 10, 100])  # 多个可组成列表
    grid_search = GridSearchCV(pipe, param_grid=param_grid)
    print(grid_search)

    # 生成训练集并输出最佳参数
    from sklearn.datasets import samples_generator
    X, y = samples_generator.make_classification(
        n_features=20, n_informative=3, n_redundant=0, n_classes=4,
        n_clusters_per_class=2)
    grid_search.fit(X, y)
    print(grid_search.best_params_)


def plot_roc():
    """
    多分类问题，每一类的roc曲线及整体宏微平均（本例自己计算，实际可以利用 roc_auc_score）
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from scipy import interp

    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 多分类问题
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                             random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 计算微平均
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ##############################################################################
    # Plot of a ROC curve for a specific class
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    ##############################################################################
    # Plot ROC curves for the multiclass problem

    # Compute macro-average ROC curve and ROC area

    # 所有的误警率
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # 对正警率进行插值
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def plot_grid_search_digits():
    """
    网格搜索手写字的识别的最佳模型参数
    1、GridSearchCV 的参数及属性
    2、GridSearchCV 中的score可以是多个
    2.1 cv_results_属性内是评价, 可以转化成pandas DataFrame
        df = pd.pd.DataFrame(clf.cv_results_)
        writer = pd.ExcelWriter('C:\\Users\\Administrator\\Desktop\\output.xlsx')
        df.to_excel(writer, 'Sheet1')
        writer.save()
    2.2 best_params_会给出当前评分中的最佳评分的参数组合

    3、利用classification_report 可以得出相同结论（precision    recall  f1-score   support）
    """
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.svm import SVC

    digits = datasets.load_digits()

    # To apply an classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)

    # 设置搜索的超参数取值范围
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score)
        # clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring=['precision_macro', 'recall_macro'],
        #                    refit='precision_macro', return_train_score=True)
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


if __name__ == '__main__':
    pass
