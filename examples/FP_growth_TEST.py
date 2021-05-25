"""
@ Filename:       FP_growth_TEST.py
@ Author:         Ryuk
@ Create Date:    2019-05-30   
@ Update Date:    2019-05-31 
@ Description:    Implement FP_growth_TEST
"""

from AssociationAnalysis import FPgrowth
import numpy as np
import pandas as pd
import time

trainData = [['bread', 'milk', 'vegetable', 'fruit', 'eggs'],
           ['noodle', 'beef', 'pork', 'water', 'socks', 'gloves', 'shoes', 'rice'],
           ['socks', 'gloves'],
           ['bread', 'milk', 'shoes', 'socks', 'eggs'],
           ['socks', 'shoes', 'sweater', 'cap', 'milk', 'vegetable', 'gloves'],
           ['eggs', 'bread', 'milk', 'fish', 'crab', 'shrimp', 'rice']]

time_start1 = time.time()
clf1 = FPgrowth()
pred1 = clf1.train(trainData)
time_end1 = time.time()
print("Runtime of FP-growth:", time_end1-time_start1)
