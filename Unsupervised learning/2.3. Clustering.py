"""
===================
Clustering (聚类)
==================
k-means：
    Batch K-Means:加快训练速度，但是稍逊于传统kmeans

Affinity propagation 聚类:
    在相似性矩阵上，算法通过在数据点之间传递信息
        （responsibility和availability，前者决定点i有多大意愿选择k作为自己的代表examplar，
        后者决定k有多大意愿决定呗i选择做代表），不断修改聚类中心的数量和位置，直到整个数据的
        net similarity（聚类中心k自己对自己的similarity+所有节点i!=k到k的similarity）达到最大。
    https://www.cnblogs.com/huadongw/p/4202492.html
    http://wiki.swarma.net/index.php/Affinity_propagation_%E8%81%9A%E7%B1%BB

Mean Shift:
    质心每次在超球体内偏移,超球的半径确定：estimate_bandwidth()
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.76.8968&rep=rep1&type=pdf
    http://www.cnblogs.com/liqizhou/archive/2012/05/12/2497220.html

谱聚类（spectral clustering）:
    它的主要思想是把所有的数据看做空间中的点，这些点之间可以用边连接起来。距离较远
        的两个点之间的边权重值较低，而距离较近的两个点之间的边权重值较高，通过对所
        有数据点组成的图进行切图，让切图后不同的子图间边权重和尽可能的低，而子图内
        的边权重和尽可能的高，从而达到聚类的目的。

    谱聚类算法的主要优点有：
　　　　1）谱聚类只需要数据之间的相似度矩阵，因此对于处理稀疏数据的聚类很有效。这点传统聚类算法比如K-Means很难做到
　　　　2）由于使用了降维，因此在处理高维数据聚类时的复杂度比传统聚类算法好。

　　谱聚类算法的主要缺点有：
　　　　1）如果最终聚类的维度非常高，则由于降维的幅度不够，谱聚类的运行速度和最后的聚类效果均不好。
　　　　2) 聚类效果依赖于相似矩阵，不同的相似矩阵得到的最终聚类效果可能很不同。

    https://www.cnblogs.com/pinard/p/6221564.html

层次聚类（Hierarchical clustering）：
    试图在不同层次对数据集进行划分，从而形成树形的聚类结构。（自底向上，自顶向下）：
        Ward minimizes the sum of squared differences within all clusters. It is a variance-minimizing approach and in this sense is similar to the k-means objective function but tackled with an agglomerative hierarchical approach.
        Maximum or complete linkage minimizes the maximum distance between observations of pairs of clusters.
        Average linkage minimizes the average of the distances between all observations of pairs of clusters.
        Single linkage minimizes the distance between the closest observations of pairs of clusters.
        （以上不同判定距离的链接方法）

    变体-BIRCH算法（遍历一遍，适合大数据集，调参复杂）：
        BIRCH算法利用了一个树结构来帮助我们快速的聚类，这个数结构类似于平衡B+树，
        一般将它称之为聚类特征树(Clustering Feature Tree，简称CF Tree)。这颗树
        的每一个节点是由若干个聚类特征(Clustering Feature，简称CF)组成。
        https://www.cnblogs.com/pinard/p/6179132.html

密度聚类：
    （DBSCAN）：邻域，核心对象，密度直达，密度可达，密度相连


评价指标：
    =========== ========================================================
    Shorthand    full name
    =========== ========================================================
    homo         homogeneity score
    compl        completeness score
    v-meas       V measure
    ARI          adjusted Rand index
    AMI          adjusted mutual information
    silhouette   silhouette coefficient
    =========== ========================================================

"""


def plot_kmeans_assumptions():
    """
    ====================================
    Demonstration of k-means assumptions
    ====================================
    k-means对于不同类数据，或不同参数的聚类结果，有一定缺陷

    This example is meant to illustrate situations where k-means will produce
    unintuitive and possibly unexpected clusters. In the first three plots, the
    input data does not conform to some implicit assumption that k-means makes and
    undesirable clusters are produced as a result. In the last plot, k-means
    returns intuitive clusters despite unevenly sized blobs.
    """
    print(__doc__)

    # Author: Phil Roth <mr.phil.roth@gmail.com>
    # License: BSD 3 clause

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    plt.figure(figsize=(12, 12))

    n_samples = 1500
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)

    # Incorrect number of clusters
    y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

    plt.subplot(221)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("Incorrect Number of Blobs")

    # Anisotropicly distributed data
    transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    X_aniso = np.dot(X, transformation)
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

    plt.subplot(222)
    plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
    plt.title("Anisotropicly Distributed Blobs")

    # Different variance
    X_varied, y_varied = make_blobs(n_samples=n_samples,
                                    cluster_std=[1.0, 2.5, 0.5],
                                    random_state=random_state)
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

    plt.subplot(223)
    plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
    plt.title("Unequal Variance")

    # Unevenly sized blobs
    X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
    y_pred = KMeans(n_clusters=3,
                    random_state=random_state).fit_predict(X_filtered)

    plt.subplot(224)
    plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
    plt.title("Unevenly Sized Blobs")

    plt.show()


def plot_kmeans_digits():
    """
    ===========================================================
    A demo of K-Means clustering on the handwritten digits data
    ===========================================================

    In this example we compare the various initialization strategies for
    K-means in terms of runtime and quality of the results.

    As the ground truth is known here, we also apply different cluster
    quality metrics to judge the goodness of fit of the cluster labels to the
    ground truth.

    Cluster quality metrics evaluated (see :ref:`clustering_evaluation` for
    definitions and discussions of the metrics):

    =========== ========================================================
    Shorthand    full name
    =========== ========================================================
    homo         homogeneity score
    compl        completeness score
    v-meas       V measure
    ARI          adjusted Rand index
    AMI          adjusted mutual information
    silhouette   silhouette coefficient
    =========== ========================================================

    """
    print(__doc__)

    from time import time
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import metrics
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import scale

    np.random.seed(42)

    digits = load_digits()
    data = scale(digits.data)

    n_samples, n_features = data.shape
    n_digits = len(np.unique(digits.target))
    labels = digits.target

    sample_size = 300

    print("n_digits: %d, \t n_samples %d, \t n_features %d"
          % (n_digits, n_samples, n_features))

    print(82 * '_')
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

    def bench_k_means(estimator, name, _data):
        t0 = time()
        estimator.fit(_data)
        print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
              % (name, (time() - t0), estimator.inertia_,
                 metrics.homogeneity_score(labels, estimator.labels_),
                 metrics.completeness_score(labels, estimator.labels_),
                 metrics.v_measure_score(labels, estimator.labels_),
                 metrics.adjusted_rand_score(labels, estimator.labels_),
                 metrics.adjusted_mutual_info_score(labels, estimator.labels_),
                 metrics.silhouette_score(_data, estimator.labels_,
                                          metric='euclidean',
                                          sample_size=sample_size)))

    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
                  name="k-means++", _data=data)

    bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
                  name="random", _data=data)

    # in this case the seeding of the centers is deterministic, hence we run the
    # kmeans algorithm only once with n_init=1
    pca = PCA(n_components=n_digits).fit(data)
    bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
                  name="PCA-based",
                  _data=data)
    print(82 * '_')

    # #############################################################################
    # Visualize the results on PCA-reduced data

    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def plot_mini_batch_kmeans():
    """
    ====================================================================
    Comparison of the K-Means and MiniBatchKMeans clustering algorithms
    ====================================================================
    画图技巧

    We want to compare the performance of the MiniBatchKMeans and KMeans:
    the MiniBatchKMeans is faster, but gives slightly different results (see
    :ref:`mini_batch_kmeans`).

    We will cluster a set of data, first with KMeans and then with
    MiniBatchKMeans, and plot the results.
    We will also plot the points that are labelled differently between the two
    algorithms.
    """
    print(__doc__)

    import time

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.cluster import MiniBatchKMeans, KMeans
    from sklearn.metrics.pairwise import pairwise_distances_argmin
    from sklearn.datasets.samples_generator import make_blobs

    # #############################################################################
    # Generate sample data
    np.random.seed(0)

    batch_size = 45
    centers = [[1, 1], [-1, -1], [1, -1]]
    n_clusters = len(centers)
    X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

    # #############################################################################
    # Compute clustering with Means

    k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
    t0 = time.time()
    k_means.fit(X)
    t_batch = time.time() - t0

    # #############################################################################
    # Compute clustering with MiniBatchKMeans

    mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=batch_size,
                          n_init=10, max_no_improvement=10, verbose=0)
    t0 = time.time()
    mbk.fit(X)
    t_mini_batch = time.time() - t0

    # #############################################################################
    # Plot result

    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']

    # We want to have the same colors for the same cluster from the
    # MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
    # closest one.
    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
    mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)
    order = pairwise_distances_argmin(k_means_cluster_centers,
                                      mbk_means_cluster_centers)

    # KMeans
    ax = fig.add_subplot(1, 3, 1)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)
    ax.set_title('KMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' % (
        t_batch, k_means.inertia_))

    # MiniBatchKMeans
    ax = fig.add_subplot(1, 3, 2)
    for k, col in zip(range(n_clusters), colors):
        my_members = mbk_means_labels == order[k]
        cluster_center = mbk_means_cluster_centers[order[k]]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=6)
    ax.set_title('MiniBatchKMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %
             (t_mini_batch, mbk.inertia_))

    # Initialise the different array to all False
    different = (mbk_means_labels == 4)
    ax = fig.add_subplot(1, 3, 3)

    for k in range(n_clusters):
        different += ((k_means_labels == k) != (mbk_means_labels == order[k]))

    identic = np.logical_not(different)
    ax.plot(X[identic, 0], X[identic, 1], 'w',
            markerfacecolor='#bbbbbb', marker='.')
    ax.plot(X[different, 0], X[different, 1], 'w',
            markerfacecolor='m', marker='.')
    ax.set_title('Difference')
    ax.set_xticks(())
    ax.set_yticks(())

    plt.show()


def plot_dict_face_patches():
    """
    Online learning of a dictionary of parts of faces
    ==================================================

    This example uses a large dataset of faces to learn a set of 20 x 20
    images patches that constitute faces.

    From the programming standpoint, it is interesting because it shows how
    to use the online API of the scikit-learn to process a very large
    dataset by chunks. The way we proceed is that we load an image at a time
    and extract randomly 50 patches from this image. Once we have accumulated
    500 of these patches (using 10 images), we run the `partial_fit` method
    of the online KMeans object, MiniBatchKMeans.

    The verbose setting on the MiniBatchKMeans enables us to see that some
    clusters are reassigned during the successive calls to
    partial-fit. This is because the number of patches that they represent
    has become too low, and it is better to choose a random new
    cluster.
    """
    print(__doc__)

    import time

    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn import datasets
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.image import extract_patches_2d

    faces = datasets.fetch_olivetti_faces()

    # #############################################################################
    # Learn the dictionary of images

    print('Learning the dictionary... ')
    rng = np.random.RandomState(0)
    kmeans = MiniBatchKMeans(n_clusters=81, random_state=rng, verbose=True)
    patch_size = (20, 20)

    buffer = []
    t0 = time.time()

    # The online learning part: cycle over the whole dataset 6 times
    index = 0
    for _ in range(6):
        for img in faces.images:
            data = extract_patches_2d(img, patch_size, max_patches=50,
                                      random_state=rng)
            data = np.reshape(data, (len(data), -1))
            buffer.append(data)
            index += 1
            if index % 10 == 0:
                data = np.concatenate(buffer, axis=0)
                data -= np.mean(data, axis=0)
                data /= np.std(data, axis=0)
                kmeans.partial_fit(data)
                buffer = []
            if index % 100 == 0:
                print('Partial fit of %4i out of %i'
                      % (index, 6 * len(faces.images)))

    dt = time.time() - t0
    print('done in %.2fs.' % dt)

    # #############################################################################
    # Plot the results
    plt.figure(figsize=(4.2, 4))
    for i, patch in enumerate(kmeans.cluster_centers_):
        plt.subplot(9, 9, i + 1)
        plt.imshow(patch.reshape(patch_size), cmap=plt.cm.gray,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())

    plt.suptitle('Patches of faces\nTrain time %.1fs on %d patches' %
                 (dt, 8 * len(faces.images)), fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    plt.show()


def plot_affinity_propagation():
    """
    =================================================
    Demo of affinity propagation clustering algorithm
    =================================================

    Reference:
    Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
    Between Data Points", Science Feb. 2007

    """
    print(__doc__)

    from sklearn.cluster import AffinityPropagation
    from sklearn import metrics
    from sklearn.datasets.samples_generator import make_blobs

    # #############################################################################
    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
                                random_state=0)

    # #############################################################################
    # Compute Affinity Propagation
    af = AffinityPropagation(preference=-50).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt
    from itertools import cycle

    plt.close('all')
    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X[cluster_centers_indices[k]]
        plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        for x in X[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def plot_mean_shift():
    """
    =============================================
    A demo of the mean-shift clustering algorithm
    =============================================

    Reference:

    Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
    feature space analysis". IEEE Transactions on Pattern Analysis and
    Machine Intelligence. 2002. pp. 603-619.

    """
    print(__doc__)

    import numpy as np
    from sklearn.cluster import MeanShift, estimate_bandwidth
    from sklearn.datasets.samples_generator import make_blobs

    # #############################################################################
    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

    # #############################################################################
    # Compute clustering with MeanShift

    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt
    from itertools import cycle

    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def plot_segmentation_toy():
    """
    ===========================================
    Spectral clustering for image segmentation
    ===========================================

    In this example, an image with connected circles is generated and
    spectral clustering is used to separate the circles.

    In these settings, the :ref:`spectral_clustering` approach solves the problem
    know as 'normalized graph cuts': the image is seen as a graph of
    connected voxels, and the spectral clustering algorithm amounts to
    choosing graph cuts defining regions while minimizing the ratio of the
    gradient along the cut, and the volume of the region.

    As the algorithm tries to balance the volume (ie balance the region
    sizes), if we take circles with different sizes, the segmentation fails.

    In addition, as there is no useful information in the intensity of the image,
    or its gradient, we choose to perform the spectral clustering on a graph
    that is only weakly informed by the gradient. This is close to performing
    a Voronoi partition of the graph.

    In addition, we use the mask of the objects to restrict the graph to the
    outline of the objects. In this example, we are interested in
    separating the objects one from the other, and not from the background.
    """
    print(__doc__)

    # Authors:  Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
    #           Gael Varoquaux <gael.varoquaux@normalesup.org>
    # License: BSD 3 clause

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.feature_extraction import image
    from sklearn.cluster import spectral_clustering

    l = 100
    x, y = np.indices((l, l))

    center1 = (28, 24)
    center2 = (40, 50)
    center3 = (67, 58)
    center4 = (24, 70)

    radius1, radius2, radius3, radius4 = 16, 14, 15, 14

    circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
    circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
    circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2
    circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius4 ** 2

    # #############################################################################
    # 4 circles
    img = circle1 + circle2 + circle3 + circle4

    # We use a mask that limits to the foreground: the problem that we are
    # interested in here is not separating the objects from the background,
    # but separating them one from the other.
    mask = img.astype(bool)

    img = img.astype(float)
    img += 1 + 0.2 * np.random.randn(*img.shape)

    # Convert the image into a graph with the value of the gradient on the
    # edges.
    graph = image.img_to_graph(img, mask=mask)

    # Take a decreasing function of the gradient: we take it weakly
    # dependent from the gradient the segmentation is close to a voronoi
    graph.data = np.exp(-graph.data / graph.data.std())

    # Force the solver to be arpack, since amg is numerically
    # unstable on this example
    labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')
    label_im = np.full(mask.shape, -1.)
    label_im[mask] = labels

    plt.matshow(img)
    plt.matshow(label_im)

    # #############################################################################
    # 2 circles
    img = circle1 + circle2
    mask = img.astype(bool)
    img = img.astype(float)

    img += 1 + 0.2 * np.random.randn(*img.shape)

    graph = image.img_to_graph(img, mask=mask)
    graph.data = np.exp(-graph.data / graph.data.std())

    labels = spectral_clustering(graph, n_clusters=2, eigen_solver='arpack')
    label_im = np.full(mask.shape, -1.)
    label_im[mask] = labels

    plt.matshow(img)
    plt.matshow(label_im)

    plt.show()


def plot_dbscan():
    """
    ===================================
    Demo of DBSCAN clustering algorithm
    ===================================

    Finds core samples of high density and expands clusters from them.

    """
    print(__doc__)

    import numpy as np

    from sklearn.cluster import DBSCAN
    from sklearn import metrics
    from sklearn.datasets.samples_generator import make_blobs
    from sklearn.preprocessing import StandardScaler

    # #############################################################################
    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                random_state=0)

    X = StandardScaler().fit_transform(X)

    # #############################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def plot_cluster_comparison():
    """
    =========================================================
    Comparing different clustering algorithms on toy datasets
    =========================================================

    This example shows characteristics of different
    clustering algorithms on datasets that are "interesting"
    but still in 2D. With the exception of the last dataset,
    the parameters of each of these dataset-algorithm pairs
    has been tuned to produce good clustering results. Some
    algorithms are more sensitive to parameter values than
    others.

    The last dataset is an example of a 'null' situation for
    clustering: the data is homogeneous, and there is no good
    clustering. For this example, the null dataset uses the
    same parameters as the dataset in the row above it, which
    represents a mismatch in the parameter values and the
    data structure.

    While these examples give some intuition about the
    algorithms, this intuition might not apply to very high
    dimensional data.
    """
    print(__doc__)

    import time
    import warnings

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import cluster, datasets, mixture
    from sklearn.neighbors import kneighbors_graph
    from sklearn.preprocessing import StandardScaler
    from itertools import cycle, islice

    np.random.seed(0)

    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============
    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                          noise=.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(n_samples=n_samples,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=random_state)

    # ============
    # Set up cluster parameters
    # ============
    plt.figure(figsize=(9 * 2 + 3, 12.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)

    plot_num = 1

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 3}

    datasets = [
        (noisy_circles, {'damping': .77, 'preference': -240,
                         'quantile': .2, 'n_clusters': 2}),
        (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
        (varied, {'eps': .18, 'n_neighbors': 2}),
        (aniso, {'eps': .15, 'n_neighbors': 2}),
        (blobs, {}),
        (no_structure, {})]

    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        X, y = dataset

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)

        # estimate bandwidth for mean shift
        bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

        # connectivity matrix for structured Ward
        connectivity = kneighbors_graph(
            X, n_neighbors=params['n_neighbors'], include_self=False)
        # make connectivity symmetric
        connectivity = 0.5 * (connectivity + connectivity.T)

        # ============
        # Create cluster objects
        # ============
        ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
        ward = cluster.AgglomerativeClustering(
            n_clusters=params['n_clusters'], linkage='ward',
            connectivity=connectivity)
        spectral = cluster.SpectralClustering(
            n_clusters=params['n_clusters'], eigen_solver='arpack',
            affinity="nearest_neighbors")
        dbscan = cluster.DBSCAN(eps=params['eps'])
        affinity_propagation = cluster.AffinityPropagation(
            damping=params['damping'], preference=params['preference'])
        average_linkage = cluster.AgglomerativeClustering(
            linkage="average", affinity="cityblock",
            n_clusters=params['n_clusters'], connectivity=connectivity)
        birch = cluster.Birch(n_clusters=params['n_clusters'])
        gmm = mixture.GaussianMixture(
            n_components=params['n_clusters'], covariance_type='full')

        clustering_algorithms = (
            ('MiniBatchKMeans', two_means),
            ('AffinityPropagation', affinity_propagation),
            ('MeanShift', ms),
            ('SpectralClustering', spectral),
            ('Ward', ward),
            ('AgglomerativeClustering', average_linkage),
            ('DBSCAN', dbscan),
            ('Birch', birch),
            ('GaussianMixture', gmm)
        )

        for name, algorithm in clustering_algorithms:
            t0 = time.time()

            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                            "connectivity matrix is [0-9]{1,2}" +
                            " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                            " may not work as expected.",
                    category=UserWarning)
                algorithm.fit(X)

            t1 = time.time()
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(X)

            plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
            if i_dataset == 0:
                plt.title(name, size=18)

            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.xticks(())
            plt.yticks(())
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plot_num += 1

    plt.show()


if __name__ == '__main__':
    plot_segmentation_toy()
