"""
@ Filename:       HMM_TEST.py
@ Author:         Danc1elion
@ Create Date:    2019-06-12   
@ Update Date:    2019-06-16
@ Description:    Implement HMM_TEST
"""

from HMM import HiddenMarkovModel
import numpy as np
import time

Q = np.array([0, 1]) # hot 0, cold 1
V = np.array([0, 1, 2])
O = np.array([[2, 2, 1], [0, 0, 1], [0, 1, 2]])
I = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 0]])
test = np.array([0, 1, 2])

# # supervised learning algorithm
time_start1 = time.time()
clf1 = HiddenMarkovModel(Q, V)
clf1.train(O, I)
time_end1 = time.time()
print("Supervised learning parameters:")
print("Transfer probability  matrix\n", clf1.A)
print("Observation probability  matirx\n", clf1.B)
print("Initial state probability \n", clf1.Pi)
print("Prediction of Supervised learning", clf1.predict(test))
print("Runtime of Supervised learning:", time_end1-time_start1)
print("________________BOUNDARY_______________________________________")
# unsupervised learning algorithm
time_start2 = time.time()
clf2 = HiddenMarkovModel(Q, V)
clf2.train(O)
time_end2 = time.time()
print("Unsupervised learning  parameters:")
print("Transfer probability  matrix\n", clf2.A)
print("Observation probability  matirx\n", clf2.B)
print("Initial state probability \n", clf2.Pi)
print("Prediction of Unsupervised learning", clf2.predict(test))
print("Runtime of Unsupervised learning:", time_end2-time_start2)
