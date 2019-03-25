"""
============================
Gaussian Processes（高斯过程）
============================
https://scikit-learn.org/stable/modules/gaussian_process.html

算法过程没看懂

高斯过程的优点是：
    预测内插了观察结果（至少对于正则核）。
    预测是概率（Gaussian），所以人们可以计算经验置信区间，并根据那些应该在某些兴趣区域改编
        （在线拟合，自适应）预测的那些来确定。
    通用性：可以指定不同的内核。提供通用内核，但也可以指定自定义内核。

高斯过程的缺点包括：
    它们不稀疏，即他们使用整个样本/特征信息来执行预测。
    它们在高维空间中失去效率 - 即当功能数量超过几十个时。
"""