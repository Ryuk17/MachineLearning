"""
@ Filename:       PCA_TEST.py
@ Author:         Danc1elion
@ Create Date:    2019-06-03   
@ Update Date:    2019-06-03 
@ Description:    Implement PCA_TEST
"""

from DimensionReduction import PCA
import numpy as np
from sklearn.decomposition import PCA as pca
import time

data = np.array([[2.5, 2.4],
                [0.5, 0.7],
                [2.2, 2.9],
                [1.9, 2.2],
                [3.1, 3.0],
                [2.3, 2.7],
                [2, 1.6],
                [1, 1.1],
                [1.5,1.6],
                [1.1, 0.9]])
time_start1 = time.time()
clf1 = PCA()
clf1.train(data)
print(clf1.transformData(data))
time_end1 = time.time()
print("Runtime of PCA:", time_end1-time_start1)

time_start2 = time.time()
clf1 = pca(1)
x = clf1.fit_transform(data)
print(x)
time_end2 = time.time()
print("Runtime of sklearn PCA:", time_end2-time_start2)
