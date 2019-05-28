"""
@ Filename:       Apriori_TEST.py
@ Author:         Danc1elion
@ Create Date:    2019-05-28   
@ Update Date:    2019-05-28 
@ Description:    Implement Apriori_TEST
"""

from AssociationAnalysis import Apriori
import numpy as np
import pandas as pd
import time

trainData = pd.read_table('../dataset/dataset7/train.txt', header=None,encoding='gb2312', delim_whitespace=True)
trainData = np.array(trainData)

time_start1 = time.time()
clf1 = Apriori()
pred1 = clf1.train(trainData)
time_end1 = time.time()
print("Runtime of Apriori:", time_end1-time_start1)
