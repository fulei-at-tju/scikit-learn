"""
=======================================================
Multiclass and multilabel algorithms(多分类及多标签算法)
=======================================================
sk-learn中所有算法都支持多分类

多分类：
    指具有两类以上的分类任务，二分类，多分类。每个样例仅有一个分类标签

多标签：
    为每个样本分配一组目标标签。
    预测不相互排斥的数据点的属性，例如与文档相关的主题。一个文字可能是宗教，政治，金融或教育的任何一个或者同时有或没有一个

Inherently multiclass:
    sklearn.naive_bayes.BernoulliNB
    sklearn.tree.DecisionTreeClassifier
    sklearn.tree.ExtraTreeClassifier
    sklearn.ensemble.ExtraTreesClassifier
    sklearn.naive_bayes.GaussianNB
    sklearn.neighbors.KNeighborsClassifier
    sklearn.semi_supervised.LabelPropagation
    sklearn.semi_supervised.LabelSpreading
    sklearn.discriminant_analysis.LinearDiscriminantAnalysis
    sklearn.svm.LinearSVC (setting multi_class=”crammer_singer”)
    sklearn.linear_model.LogisticRegression (setting multi_class=”multinomial”)
    sklearn.linear_model.LogisticRegressionCV (setting multi_class=”multinomial”)
    sklearn.neural_network.MLPClassifier
    sklearn.neighbors.NearestCentroid
    sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis
    sklearn.neighbors.RadiusNeighborsClassifier
    sklearn.ensemble.RandomForestClassifier
    sklearn.linear_model.RidgeClassifier
    sklearn.linear_model.RidgeClassifierCV
Multiclass as One-Vs-One:
    sklearn.svm.NuSVC
    sklearn.svm.SVC.
    sklearn.gaussian_process.GaussianProcessClassifier (setting multi_class = “one_vs_one”)
    Multiclass as One-Vs-All:
    sklearn.ensemble.GradientBoostingClassifier
    sklearn.gaussian_process.GaussianProcessClassifier (setting multi_class = “one_vs_rest”)
    sklearn.svm.LinearSVC (setting multi_class=”ovr”)
    sklearn.linear_model.LogisticRegression (setting multi_class=”ovr”)
    sklearn.linear_model.LogisticRegressionCV (setting multi_class=”ovr”)
    sklearn.linear_model.SGDClassifier
    sklearn.linear_model.Perceptron
    sklearn.linear_model.PassiveAggressiveClassifier
Support multilabel:
    sklearn.tree.DecisionTreeClassifier
    sklearn.tree.ExtraTreeClassifier
    sklearn.ensemble.ExtraTreesClassifier
    sklearn.neighbors.KNeighborsClassifier
    sklearn.neural_network.MLPClassifier
    sklearn.neighbors.RadiusNeighborsClassifier
    sklearn.ensemble.RandomForestClassifier
    sklearn.linear_model.RidgeClassifierCV
Support multiclass-multioutput:
    sklearn.tree.DecisionTreeClassifier
    sklearn.tree.ExtraTreeClassifier
    sklearn.ensemble.ExtraTreesClassifier
    sklearn.neighbors.KNeighborsClassifier
    sklearn.neighbors.RadiusNeighborsClassifier
    sklearn.ensemble.RandomForestClassifier
"""


def Multilabel_classification_format():
    from sklearn.preprocessing import MultiLabelBinarizer
    y = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]
    print(MultiLabelBinarizer().fit_transform(y))


if __name__ == '__main__':
    Multilabel_classification_format()
