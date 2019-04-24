from sklearn.neighbors import KNeighborsClassifier
from KNN import *
import numpy as np
import pandas as pd
import time

trainData = pd.read_table('./dataset1/train.txt',header=None,encoding='gb2312',delim_whitespace=True)
testData = pd.read_table('./dataset1/test.txt',header=None,encoding='gb2312',delim_whitespace=True)
trainLabel = np.array(trainData.pop(3))
trainData = np.array(trainData)
testLabel = np.array(testData.pop(3))
testData = np.array(testData)

time_start1 = time.time()
clf1 = KNNClassifier(k=6)
clf1.train(trainData, trainLabel)
clf1.predict(testData)
score1 = clf1.showDetectionResult(testData, testLabel)
time_end1 = time.time()
print("Accuracy of self-KNN: %f" % score1)
print("Runtime of self-KNN:", time_end1-time_start1)

time_start = time.time()
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(trainData, trainLabel)
knn.predict(testData)
score = knn.score(testData, testLabel, sample_weight=None)
time_end = time.time()
print("Accuracy of sklearn-KNN: %f" % score)
print("Runtime of sklearn-KNN:", time_end-time_start)
