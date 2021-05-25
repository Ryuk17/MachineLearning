"""
@ Filename:       RandomForest_TEST.py
@ Author:         Ryuk
@ Create Date:    2019-07-10   
@ Update Date:    2019-07-10 
@ Description:    Implement RandomForest_TEST
"""
from RandomForest import RandomForestClassifier, RandomForestRegression
import numpy as np
import pandas as pd
import time
from DecisionTree import *

trainData = pd.read_table('../dataset/dataset1/train.txt', header=None, encoding='gb2312', delim_whitespace=True)
testData = pd.read_table('../dataset/dataset1/test.txt', header=None, encoding='gb2312', delim_whitespace=True)
trainLabel = np.array(trainData.pop(3))
trainData = np.array(trainData)
testLabel = np.array(testData.pop(3))
testData = np.array(testData)

time_start1 = time.time()
clf1 = DecisionTreeClassifier()
clf1.train(trainData, trainLabel)
clf1.predict(testData)
score1 = clf1.accuarcy(testLabel)
time_end1 = time.time()
print("Accuracy of self-DecisionTree: %f" % score1)
print("Runtime of self-DecisionTree:", time_end1-time_start1)

time_start = time.time()
clf = RandomForestClassifier()
clf.train(trainData, trainLabel)
clf.predict(testData)
score = clf.accuarcy(testLabel)
time_end = time.time()
print("Accuracy of RandomForest: %f" % score)
print("Runtime of RandomForest:", time_end-time_start)


