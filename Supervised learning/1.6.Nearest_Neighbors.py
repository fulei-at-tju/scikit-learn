"""
==========================
Nearest Neighbors (最近邻)

可分类(neighbors.NearestNeighbors)，可回归 (neighbors.KNeighborsRegressor)

The choice of neighbors search algorithm is controlled through the keyword 'algorithm',
which must be one of ['auto', 'ball_tree', 'kd_tree', 'brute']
==========================
"""


def finding_the_nearest_nearest_neighbors():
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    print(distances)
    print(indices)
    A = nbrs.kneighbors_graph(X, mode='distance')  # 邻接矩阵
    print(A.toarray())


def plot_classification():
    """
    ================================
    Nearest Neighbors Supervised learning
    ================================

    Sample usage of Nearest Neighbors classification.
    It will plot the decision boundaries for each class.
    """
    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn import neighbors, datasets

    n_neighbors = 15

    # import some data to play with
    iris = datasets.load_iris()

    # we only take the first two features. We could avoid this ugly
    # slicing by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target

    h = .02  # step size in the mesh

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    for weights in ['uniform', 'distance']:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        clf.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (k = %i, weights = '%s')"
                  % (n_neighbors, weights))

    plt.show()


def plot_regression():
    """
    ============================
    Nearest Neighbors regression
    ============================

    Demonstrate the resolution of a regression problem
    using a k-Nearest Neighbor and the interpolation of the
    target using both barycenter and constant weights.

    """
    print(__doc__)

    # Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
    #         Fabian Pedregosa <fabian.pedregosa@inria.fr>
    #
    # License: BSD 3 clause (C) INRIA

    # #############################################################################
    # Generate sample data
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import neighbors

    np.random.seed(0)
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    T = np.linspace(0, 5, 500)[:, np.newaxis]
    y = np.sin(X).ravel()

    # Add noise to targets
    y[::5] += 1 * (0.5 - np.random.rand(8))

    # #############################################################################
    # Fit regression model
    n_neighbors = 5

    for i, weights in enumerate(['uniform', 'distance']):
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        y_ = knn.fit(X, y).predict(T)

        plt.subplot(2, 1, i + 1)
        plt.scatter(X, y, c='k', label='data')
        plt.plot(T, y_, c='g', label='prediction')
        plt.axis('tight')
        plt.legend()
        plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                    weights))

    plt.tight_layout()
    plt.show()


def plot_multioutput_face_completion():
    """
    ==============================================
    Face completion with a multi-output estimators
    ==============================================

    This example shows the use of multi-output estimator to complete images.
    The goal is to predict the lower half of a face given its upper half.

    The first column of images shows true faces. The next columns illustrate
    how extremely randomized trees, k nearest neighbors, linear
    regression and ridge regression complete the lower half of those faces.

    """
    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.datasets import fetch_olivetti_faces
    from sklearn.utils.validation import check_random_state

    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import RidgeCV

    # Load the faces datasets
    data = fetch_olivetti_faces()
    targets = data.target

    data = data.images.reshape((len(data.images), -1))
    train = data[targets < 30]
    test = data[targets >= 30]  # Test on independent people

    # Test on a subset of people
    n_faces = 5
    rng = check_random_state(4)
    face_ids = rng.randint(test.shape[0], size=(n_faces,))
    test = test[face_ids, :]

    n_pixels = data.shape[1]
    # Upper half of the faces
    X_train = train[:, :(n_pixels + 1) // 2]
    # Lower half of the faces
    y_train = train[:, n_pixels // 2:]
    X_test = test[:, :(n_pixels + 1) // 2]
    y_test = test[:, n_pixels // 2:]

    # Fit estimators
    ESTIMATORS = {
        "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32,
                                           random_state=0),
        "K-nn": KNeighborsRegressor(),
        "Linear regression": LinearRegression(),
        "Ridge": RidgeCV(),
    }

    y_test_predict = dict()
    for name, estimator in ESTIMATORS.items():
        estimator.fit(X_train, y_train)
        y_test_predict[name] = estimator.predict(X_test)

    # Plot the completed faces
    image_shape = (64, 64)

    n_cols = 1 + len(ESTIMATORS)
    plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
    plt.suptitle("Face completion with multi-output estimators", size=16)

    for i in range(n_faces):
        true_face = np.hstack((X_test[i], y_test[i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 1,
                              title="true faces")

        sub.axis("off")
        sub.imshow(true_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")

        for j, est in enumerate(sorted(ESTIMATORS)):
            completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

            if i:
                sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)

            else:
                sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j,
                                  title=est)

            sub.axis("off")
            sub.imshow(completed_face.reshape(image_shape),
                       cmap=plt.cm.gray,
                       interpolation="nearest")

    plt.show()


if __name__ == '__main__':
    finding_the_nearest_nearest_neighbors()
    plot_multioutput_face_completion()
