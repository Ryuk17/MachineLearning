"""
@ Filename:       LDA_TEST.py
@ Author:         Ryuk
@ Create Date:    2019-06-04   
@ Update Date:    2019-06-04 
@ Description:    Implement LDA_TEST
"""

from sklearn.model_selection import train_test_split
from DimensionReduction import LDA
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
from sklearn.linear_model import LogisticRegression
import pandas as pd

trainData = np.array(pd.read_table('./dataset/dataset1/train.txt', header=None, encoding='gb2312', delim_whitespace=True))
testData = np.array(pd.read_table('./dataset/dataset1/test.txt', header=None, encoding='gb2312', delim_whitespace=True))
train_y = trainData[:, -1]
train_x = np.delete(trainData, -1, axis=1)
test_y = testData[:, -1]
test_x = np.delete(testData, -1, axis=1)

time_start1 = time.time()
clf1 = LDA()
clf1.train(train_x, train_y)
train_x = clf1.transformData(train_x)
test_x = clf1.transformData(test_x)
clf = LogisticRegression()
clf.fit(train_x, train_y)
print("Accuracy of LDA:", clf.score(test_x, test_y))
time_end1 = time.time()
print("Runtime of LDA:", time_end1-time_start1)


time_start2 = time.time()
clf2 = LinearDiscriminantAnalysis(n_components=1)
train_x = clf2.fit_transform(train_x, train_y)
test_x = clf2.fit_transform(test_x, test_y)
clf = LogisticRegression()
clf.fit(train_x, train_y)
print("Accuracy of sklearn LDA:", clf.score(test_x, test_y))
time_end2 = time.time()
print("Runtime of sklearn LDA:", time_end2-time_start2)
