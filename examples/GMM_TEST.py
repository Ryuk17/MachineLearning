"""
@FileName: GMM_TEST.py
@Description: Implement GMM_TEST
@Author: Ryuk
@CreateDate: 2021/06/03
@LastEditTime: 2021/06/03
@LastEditors: Please set LastEditors
@Version: v0.1
"""

from sklearn.mixture import GaussianMixture
from GMM import *
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_blobs


X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)

time_start1 = time.time()
clf1 = GaussianMixtureModel(K=4)
pred = clf1.train(X)
time_end1 = time.time()
print("Runtime of GMM:", time_end1-time_start1)


time_start2 = time.time()
clf2 = GaussianMixture(n_components=4)
pred2 = clf2.fit_predict(X)
time_end2 = time.time()
print("Runtime of Sklearn GMM:", time_end2-time_start2)
plt.scatter(X[:, 0], X[:, 1], c=pred2)
plt.title('Sklearn GMM')
plt.show()






