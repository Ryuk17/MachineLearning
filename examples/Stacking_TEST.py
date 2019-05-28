"""
@ Filename:       Stacking_TEST.py
@ Author:         Danc1elion
@ Create Date:    2019-05-05
@ Update Date:    2019-05-05
@ Description:    Test Stacking
"""

from Stacking import *
from Perceptron import *
from Logistic import *
import numpy as np
import pandas as pd
import time

trainData = np.array(pd.read_table('../dataset3/train.txt', header=None, encoding='gb2312', delim_whitespace=True))
testData = np.array(pd.read_table('../dataset3/test.txt', header=None, encoding='gb2312', delim_whitespace=True))
trainLabel = trainData[:, -1]
trainData = np.delete(trainData, -1, axis=1)
testLabel = testData[:, -1]
testData = np.delete(testData, -1, axis=1)

clfs = [PerceptronClassifier(), PerceptronClassifier(), LogisticRegressionClassifier(), LogisticRegressionClassifier()]

time_start1 = time.time()
clf1 = StackingClassifier(classifier_set=clfs)
clf1.train(trainData, trainLabel)
clf1.predict(testData)
score1 = clf1.accuarcy(testLabel)
time_end1 = time.time()
print("Accuracy of self-Stacking: %f" % score1)
print("Runtime of self-Stacking:", time_end1-time_start1)

time_start2 = time.time()
clf2 = LogisticRegressionClassifier()
clf2.train(trainData, trainLabel)
clf2.predict(testData)
score2 = clf2.accuarcy(testLabel)
time_end2 = time.time()
print("Accuracy of self-Logistic: %f" % score2)
print("Runtime of self-Logistic:", time_end2-time_start2)

time_start3 = time.time()
clf3 = PerceptronClassifier()
clf3.train(trainData, trainLabel)
clf3.predict(testData)
score3 = clf3.accuarcy(testLabel)
time_end3 = time.time()
print("Accuracy of self-Perceptron: %f" % score3)
print("Runtime of self-Perceptron:", time_end3-time_start3)


