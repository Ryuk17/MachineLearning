"""
@ Filename:       HMM_TEST.py
@ Author:         Danc1elion
@ Create Date:    2019-06-12   
@ Update Date:    2019-06-15
@ Description:    Implement HMM_TEST
"""

from HMM import HiddenMarkovModel
import numpy as np

Q = np.array([0, 1]) # hot 0, cold 1
V = np.array([0, 1, 2])
O = np.array([[2, 2, 1], [0, 0, 1], [0, 1, 2]])
I = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 0]])


# # supervised learning algorithm
clf1 = HiddenMarkovModel(Q, V)
clf1.train(O, I)
print("Supervised learning  parameters:")
print("Transfer probability  matrix\n", clf1.A)
print("Observation probability  matirx\n", clf1.B)
print("Initial state probability \n", clf1.Pi)

# unsupervised learning algorithm
clf2 = HiddenMarkovModel(Q, V)
clf2.train(O)
print("Unsupervised learning  parameters:")
print("Transfer probability  matrix\n", clf2.A)
print("Observation probability  matirx\n", clf2.B)
print("Initial state probability \n", clf2.Pi)





