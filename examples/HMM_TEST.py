"""
@ Filename:       HMM_TEST.py
@ Author:         Danc1elion
@ Create Date:    2019-06-12   
@ Update Date:    2019-06-12 
@ Description:    Implement HMM_TEST
"""

from HMM import HiddenMarkovModel
import numpy as np

# supervised learning algorithm
Q = np.array([0, 1]) # hot 0, cold 1
V = np.array([1, 2, 3])
O = np.array([[3, 3, 2], [1, 1, 2], [1, 2, 3]])
I = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 0]])
clf = HiddenMarkovModel(Q, V)
clf.train(I, O)
print("Supervised learning  parameters:")
print("Transfer probability  matrix\n", clf.A)
print("Observation probability  matirx\n", clf.B)
print("Initial state probability \n", clf.Pi)
