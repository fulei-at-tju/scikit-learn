"""
===========================================
4.9. Transforming the prediction target (y)
===========================================
y值转换
preprocessing.LabelEncoder()
    transform()
    inverse_transform()
"""


def example():
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit([1, 2, 6, 4, 2])

    print(lb.classes_)
    print(lb.transform([1, 6]))

    #######################################
    lb = preprocessing.MultiLabelBinarizer()
    lb.fit_transform([(1, 2), (3,)])

    print(lb.classes_)
    ########################################
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit([1, 2, 2, 6])

    print(le.classes_)
    print(le.transform([1, 1, 2, 6]))
    print(le.inverse_transform([0, 0, 1, 2]))

    #########################################
    le = preprocessing.LabelEncoder()
    le.fit(["paris", "paris", "tokyo", "amsterdam"])

    print(list(le.classes_))
    print(le.transform(["tokyo", "tokyo", "paris"]))
    print(list(le.inverse_transform([2, 2, 1])))

