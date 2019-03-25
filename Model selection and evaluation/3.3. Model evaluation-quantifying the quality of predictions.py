"""
========================================================================
3.3. Model evaluation: quantifying the quality of predictions（模型评价）
========================================================================
各类损失的计算方法：
    https://scikit-learn.org/stable/modules/model_evaluation.html

Scoring	Function	Comment
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

越大越好：
    functions ending with _score return a value to maximize, the higher the better.
越小越好：
    functions ending with _error or _loss return a value to minimize, the lower the better.
    When converting into a scorer object using make_scorer, set the greater_is_better parameter to
    False (True by default; see the parameter description below).

二分类度量方法应用多多标签与多分类：average=None:
    ovr问题   https://blog.csdn.net/pipisorry/article/details/52574156
              https://www.jianshu.com/p/e19f557f07af
    macro:
        先对每一个类统计指标值，然后在对所有类求算术平均值。宏平均指标相对微平均指标而言受小类别的影响更大。
    micro:
        对数据集中的每一个实例不分类别进行统计建立全局混淆矩阵，然后计算相应指标。
    weighted:

metrics中的评价指标：
    Some of these are restricted to the binary classification case:
        precision_recall_curve(y_true, probas_pred)	Compute precision-recall pairs for different probability thresholds
        roc_curve(y_true, y_score[, pos_label, …])	Compute Receiver operating characteristic (ROC)
        balanced_accuracy_score(y_true, y_pred[, …])	Compute the balanced accuracy

    Others also work in the multiclass case:
        cohen_kappa_score(y1, y2[, labels, weights, …])	Cohen’s kappa: a statistic that measures inter-annotator agreement.
        confusion_matrix(y_true, y_pred[, labels, …])	Compute confusion matrix to evaluate the accuracy of a classification
        hinge_loss(y_true, pred_decision[, labels, …])	Average hinge loss (non-regularized)
        matthews_corrcoef(y_true, y_pred[, …])	Compute the Matthews correlation coefficient (MCC)

    Some also work in the multilabel case:
        accuracy_score(y_true, y_pred[, normalize, …])	Accuracy classification score.
        classification_report(y_true, y_pred[, …])	Build a text report showing the main classification metrics
        f1_score(y_true, y_pred[, labels, …])	Compute the F1 score, also known as balanced F-score or F-measure
        fbeta_score(y_true, y_pred, beta[, labels, …])	Compute the F-beta score
        hamming_loss(y_true, y_pred[, labels, …])	Compute the average Hamming loss.
        jaccard_similarity_score(y_true, y_pred[, …])	Jaccard similarity coefficient score
        log_loss(y_true, y_pred[, eps, normalize, …])	Log loss, aka logistic loss or cross-entropy loss.
        precision_recall_fscore_support(y_true, y_pred)	Compute precision, recall, F-measure and support for each class
        precision_score(y_true, y_pred[, labels, …])	Compute the precision
        recall_score(y_true, y_pred[, labels, …])	Compute the recall
        zero_one_loss(y_true, y_pred[, normalize, …])	Zero-one classification loss.

    And some work with binary and multilabel (but not multiclass) problems:
        average_precision_score(y_true, y_score[, …])	Compute average precision (AP) from prediction scores
        roc_auc_score(y_true, y_score[, average, …])	Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.


常用：--make_scorer
    accuracy_score
    auc
    classification_report:
                       precision    recall  f1-score   support
        <BLANKLINE>
             class 0       0.50      1.00      0.67         1
             class 1       0.00      0.00      0.00         1
             class 2       1.00      0.67      0.80         3
        <BLANKLINE>
           micro avg       0.60      0.60      0.60         5
           macro avg       0.50      0.56      0.49         5
        weighted avg       0.70      0.60      0.61         5
        <BLANKLINE>
    confusion_matrix
    f1_score
    fbeta_score
    log_loss
    mean_squared_error
    mean_squared_log_error
    precision_recall_curve
    precision_score
    r2_score
    recall_score
    roc_auc_score
    roc_curve
    zero_one_loss
"""

from sklearn import metrics


def examples():
    from sklearn import svm, datasets
    from sklearn.model_selection import cross_val_score
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf = svm.SVC(gamma='scale', random_state=0)
    cross_val_score(clf, X, y, scoring='recall_macro', cv=5)

    ####################################################
    from sklearn.metrics import fbeta_score, make_scorer
    ftwo_scorer = make_scorer(fbeta_score, beta=2)
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import LinearSVC
    grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=ftwo_scorer, cv=5)

    ####################################################
    # 自定义损失函数
    import numpy as np
    def my_custom_loss_func(_y_true, _y_pred):
        diff = np.abs(_y_true - _y_pred).max()
        return np.log1p(diff)

    # score will negate the return value of my_custom_loss_func,
    # which will be np.log(2), 0.693, given the values for X
    # and y defined below.
    score = make_scorer(my_custom_loss_func, greater_is_better=False)
    X = [[1], [1]]
    y = [0, 1]
    from sklearn.dummy import DummyClassifier
    clf = DummyClassifier(strategy='most_frequent', random_state=0)
    clf = clf.fit(X, y)
    print(my_custom_loss_func(clf.predict(X), y))

    print(score(clf, X, y))

    ###########################################################
    # GridSearchCV, RandomizedSearchCV and cross_validate.
    scoring = ['accuracy', 'precision']

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import make_scorer
    scoring = {'accuracy': make_scorer(accuracy_score),
               'prec': 'precision'}

    from sklearn.model_selection import cross_validate
    from sklearn.metrics import confusion_matrix
    # A sample toy binary classification dataset
    X, y = datasets.make_classification(n_classes=2, random_state=0)
    svm = LinearSVC(random_state=0)

    def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]

    def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]

    def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]

    def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
               'fp': make_scorer(fp), 'fn': make_scorer(fn)}
    cv_results = cross_validate(svm.fit(X, y), X, y,
                                scoring=scoring, cv=5)
    # Getting the test set true positive scores
    print(cv_results['test_tp'])

    # Getting the test set false negative scores
    print(cv_results['test_fn'])

    ###################################################################
    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    print(confusion_matrix(y_true, y_pred))

    ###################################################################
    import numpy as np
    from sklearn import svm
    X = np.array([[0], [1], [2], [3]])
    Y = np.array([0, 1, 2, 3])
    labels = np.array([0, 1, 2, 3])
    est = svm.LinearSVC()
    est.fit(X, Y)

    pred_decision = est.decision_function([[-1], [2], [3]])
    y_true = [0, 2, 3]


def plot_confusion_matrix():
    """
    ================
    Confusion matrix
    ================

    Example of confusion matrix usage to evaluate the quality
    of the output of a classifier on the iris data set. The
    diagonal elements represent the number of points for which
    the predicted label is equal to the true label, while
    off-diagonal elements are those that are mislabeled by the
    classifier. The higher the diagonal values of the confusion
    matrix the better, indicating many correct predictions.

    The figures show the confusion matrix with and without
    normalization by class support size (number of elements
    in each class). This kind of normalization can be
    interesting in case of class imbalance to have a more
    visual interpretation of which class is being misclassified.

    Here the results are not as good as they could be as our
    choice for the regularization parameter C was not the best.
    In real life applications this parameter is usually chosen
    using :ref:`grid_search`.

    """

    print(__doc__)

    import itertools
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import svm, datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Run classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    classifier = svm.SVC(kernel='linear', C=0.01)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()


def plot_digits_classification():
    """
    ================================
    Recognizing hand-written digits
    ================================

    An example showing how the scikit-learn can be used to recognize images of
    hand-written digits.

    This example is commented in the
    :ref:`tutorial section of the user manual <introduction>`.

    """
    print(__doc__)

    # Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
    # License: BSD 3 clause

    # Standard scientific Python imports
    import matplotlib.pyplot as plt

    # Import datasets, classifiers and performance metrics
    from sklearn import datasets, svm, metrics

    # The digits dataset
    digits = datasets.load_digits()

    # The data that we are interested in is made of 8x8 images of digits, let's
    # have a look at the first 4 images, stored in the `images` attribute of the
    # dataset.  If we were working from image files, we could load them using
    # matplotlib.pyplot.imread.  Note that each image must have the same size. For these
    # images, we know which digit they represent: it is given in the 'target' of
    # the dataset.
    images_and_labels = list(zip(digits.images, digits.target))
    for index, (image, label) in enumerate(images_and_labels[:4]):
        plt.subplot(2, 4, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: %i' % label)

    # To apply a classifier on this data, we need to flatten the image, to
    # turn the data in a (samples, feature) matrix:
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    classifier = svm.SVC(gamma=0.001)

    # We learn the digits on the first half of the digits
    classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

    # Now predict the value of the digit on the second half:
    expected = digits.target[n_samples // 2:]
    predicted = classifier.predict(data[n_samples // 2:])

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
    for index, (image, prediction) in enumerate(images_and_predictions[:4]):
        plt.subplot(2, 4, index + 5)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Prediction: %i' % prediction)

    plt.show()


def plot_roc():
    """
    =======================================
    Receiver Operating Characteristic (ROC)
    =======================================

    Example of Receiver Operating Characteristic (ROC) metric to evaluate
    classifier output quality.

    ROC curves typically feature true positive rate on the Y axis, and false
    positive rate on the X axis. This means that the top left corner of the plot is
    the "ideal" point - a false positive rate of zero, and a true positive rate of
    one. This is not very realistic, but it does mean that a larger area under the
    curve (AUC) is usually better.

    The "steepness" of ROC curves is also important, since it is ideal to maximize
    the true positive rate while minimizing the false positive rate.

    Multiclass settings
    -------------------

    ROC curves are typically used in binary classification to study the output of
    a classifier. In order to extend ROC curve and ROC area to multi-class
    or multi-label classification, it is necessary to binarize the output. One ROC
    curve can be drawn per label, but one can also draw a ROC curve by considering
    each element of the label indicator matrix as a binary prediction
    (micro-averaging).

    Another evaluation measure for multi-class classification is
    macro-averaging, which gives equal weight to the classification of each
    label.

    .. note::

        See also :func:`sklearn.metrics.roc_auc_score`,
                 :ref:`sphx_glr_auto_examples_model_selection_plot_roc_crossval.py`.

    """
    print(__doc__)

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

    # Binarize the output
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

    # Compute micro-average ROC curve and ROC area
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

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
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


if __name__ == '__main__':
    pass
