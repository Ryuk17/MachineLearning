from sklearn.linear_model import LogisticRegression
from Logistic import *
import numpy as np
import pandas as pd
import time

trainData = np.array(pd.read_table('../dataset2/train.txt',header=None,encoding='gb2312',delim_whitespace=True))
testData = np.array(pd.read_table('../dataset2/test.txt',header=None,encoding='gb2312',delim_whitespace=True))
trainLabel = trainData[:, -1]
trainData = np.delete(trainData, -1, axis=1)
testLabel = testData[:, -1]
testData = np.delete(testData, -1, axis=1)

time_start1 = time.time()
clf1 = LogisticRegressionClassifier()
clf1.train(trainData, trainLabel)
clf1.predict(testData)
score1 = clf1.accuarcy(testLabel)
time_end1 = time.time()
print("Accuracy of self-LogisticRegression: %f" % score1)
print("Runtime of self-LogisticRegression:", time_end1-time_start1)

time_start = time.time()
clf = LogisticRegression()
clf.fit(trainData, trainLabel)
clf.predict(testData)
score = clf.score(testData, testLabel, sample_weight=None)
time_end = time.time()
print("Accuracy of sklearn-LogisticRegression: %f" % score)
print("Runtime of sklearn-LogisticRegression:", time_end-time_start)
