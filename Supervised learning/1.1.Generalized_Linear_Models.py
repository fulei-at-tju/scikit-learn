"""
广义线性模型
"""
from sklearn import linear_model


def Ordinary_Least_Squares():
    """
    最小二乘法
    复杂度：O(np2)
    :return:
    """
    reg = linear_model.LinearRegression()
    reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])

    print(reg.coef_, reg.intercept_)


def plot_ols():
    """
    =========================================================
    Linear Regression Example
    =========================================================
    This example uses the only the first feature of the `diabetes` dataset, in
    order to illustrate a two-dimensional plot of this regression technique. The
    straight line can be seen in the plot, showing how linear regression attempts
    to draw a straight line that will best minimize the residual sum of squares
    between the observed responses in the dataset, and the responses predicted by
    the linear approximation.

    The coefficients, the residual sum of squares and the variance score are also
    calculated.

    """
    print(__doc__)

    # Code source: Jaques Grobler
    # License: BSD 3 clause

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import datasets, linear_model
    from sklearn.metrics import mean_squared_error, r2_score

    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()

    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]

    # Split the data into training/testing sets
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    # Split the targets into training/testing sets
    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Make predictions using the testing set
    diabetes_y_pred = regr.predict(diabetes_X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

    # Plot outputs
    plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
    plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()


def Ridge_Regression():
    """
    岭回归：对于有些矩阵，矩阵中某个元素的一个很小的变动，会引起最后计算结果误差很大，这种矩阵称为“病态矩阵”。
            有些时候不正确的计算方法也会使一个正常的矩阵在运算中表现出病态。
            岭回归是对最小二乘回归的一种补充，它损失了无偏性，来换取高的数值稳定性，从而得到较高的计算精度

            增加正则化系数
    复杂度：O(np2)
    :return:
    """
    reg = linear_model.Ridge(alpha=.5)
    reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
    print(reg.coef_, reg.intercept_)


def plot_logistic_l1_l2_sparsity():
    """
    ==============================================
    L1 Penalty and Sparsity in Logistic Regression
    ==============================================

    Comparison of the sparsity (percentage of zero coefficients) of solutions when
    L1 and L2 penalty are used for different values of C. We can see that large
    values of C give more freedom to the model.  Conversely, smaller values of C
    constrain the model more. In the L1 penalty case, this leads to sparser
    solutions.

    We classify 8x8 images of digits into two classes: 0-4 against 5-9.
    The visualization shows coefficients of the models for varying C.
    """

    print(__doc__)

    # Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
    #          Mathieu Blondel <mathieu@mblondel.org>
    #          Andreas Mueller <amueller@ais.uni-bonn.de>
    # License: BSD 3 clause

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LogisticRegression
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler

    digits = datasets.load_digits()

    X, y = digits.data, digits.target
    X = StandardScaler().fit_transform(X)

    # classify small against large digits
    y = (y > 4).astype(np.int)

    # Set regularization parameter
    for i, C in enumerate((1, 0.1, 0.01)):
        # turn down tolerance for short training time
        clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01, solver='saga')
        clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01, solver='saga')
        clf_l1_LR.fit(X, y)
        clf_l2_LR.fit(X, y)

        coef_l1_LR = clf_l1_LR.coef_.ravel()
        coef_l2_LR = clf_l2_LR.coef_.ravel()

        # coef_l1_LR contains zeros due to the
        # L1 sparsity inducing norm

        sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
        sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100

        print("C=%.2f" % C)
        print("Sparsity with L1 penalty: %.2f%%" % sparsity_l1_LR)
        print("score with L1 penalty: %.4f" % clf_l1_LR.score(X, y))
        print("Sparsity with L2 penalty: %.2f%%" % sparsity_l2_LR)
        print("score with L2 penalty: %.4f" % clf_l2_LR.score(X, y))

        l1_plot = plt.subplot(3, 2, 2 * i + 1)
        l2_plot = plt.subplot(3, 2, 2 * (i + 1))
        if i == 0:
            l1_plot.set_title("L1 penalty")
            l2_plot.set_title("L2 penalty")

        l1_plot.imshow(np.abs(coef_l1_LR.reshape(8, 8)), interpolation='nearest',
                       cmap='binary', vmax=1, vmin=0)
        l2_plot.imshow(np.abs(coef_l2_LR.reshape(8, 8)), interpolation='nearest',
                       cmap='binary', vmax=1, vmin=0)
        plt.text(-8, 3, "C = %.2f" % C)

        l1_plot.set_xticks(())
        l1_plot.set_yticks(())
        l2_plot.set_xticks(())
        l2_plot.set_yticks(())

    plt.show()


if __name__ == '__main__':
    plot_logistic_l1_l2_sparsity()
