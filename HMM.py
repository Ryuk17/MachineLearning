"""
@ Filename:       HMM.py
@ Author:         Danc1elion
@ Create Date:    2019-06-06   
@ Update Date:    2019-06-06 
@ Description:    Implement HMM
"""
import numpy as np
import preProcess

class HiddenMarkovModel:
    def __init__(self):
        self.initial_probability = None         # pi in size of N * 1
        self.transfer_matrix = None             # A in size of N * N
        self.observation_matrix = None          # B in size of N * M
        self.HMM = None

        self.P = None                           # probability  P(O|lambda)
        self.alpha = None                       # forward_probability
        self.beta = None                        # backward_probability

    '''
      Function:  calculateObservationProbability
      Description: calculate observation sequence probability  P(O|lambda)
      Input:    Q               dataType: list       description: the set of states
                A               dataType: matrix     description: transfer probability matrix
                B               dataType: matrix     description: observation probability matrix
                O               dataType: list       description: observation sequence
                T               dataType: list       description: length of observation sequence
                Pi              dataType: list       description: initial state probability
                method          dataType: string     description: forward algorithm or backward algorithm 
      Output:   forward_prob    dataType: list       description: forward  probability at each time step
                backward_prob   dataType: list       description: backward  probability at each time step
                p               dataType: float      description: observation sequence probability  P(O|lambda)
    '''
    def calculateObservationProbability(self, Q, A, B, O, T, Pi):
        N = len(Q)

        # forward algorithm
        forward_probability = []
        # calculate the initial state
        state = np.multiply(Pi, B[:, O[0]]).reshape(N, 1)
        forward_probability.append(state)
        # recursion
        t = 1
        while t < T:
            state = np.multiply(state, A)
            state = np.sum(state, axis=0)
            state = np.multiply(state, B[:, O[t]]).reshape(N, 1)
            forward_probability.append(state)
            t = t + 1

        # final result
        p = np.sum(state)

        # backward algorithm
        backward_probability = []
        # initial state
        state = np.ones([N, 1])
        backward_probability.append(state)

        # recursion
        t = T - 1
        while t > 0:
            temp = np.multiply(B[:, O[t]], A)
            temp = np.multiply(state, temp)
            state = np.sum(temp, axis=1)
            backward_probability.append(state)
            t = t - 1

        # final result
        state = np.multiply(state, np.multiply(Pi, B[:, O[0]]))
        p = np.sum(state)

        self.alpha = forward_probability
        self.beta = backward_probability
        return forward_probability, backward_probability, p

    '''
      Function:  getStateProbability
      Description: calculate the probability of state q at time t
      Input:  t     dataType: int     description: time t
              i     dataType: int     description: state i
      Output: r     dataType: float   description: the probability of state q at time t
    '''
    def getStateProbability(self, t, i):
        r = (self.alpha[t][i] * self.beta[t][i])/self.P
        return r

    '''
      Function:  getStateProbability
      Description: calculate the probability of state i at time t and state j and time t+1. In Ref[4] P179 Eq.(10.25)
      Input:  t     dataType: int     description: time t
              i     dataType: int     description: state i
              j     dataType: int     description: state j
      Output: r     dataType: float   description: the probability of state q at time t
    '''
    def getTransferProbability(self, t, i):
        r = (self.alpha[t][i] * self.beta[t][i])/self.P
        return r


    '''
      Function:  train
      Description: train the model
      Input:  transfer_matrix         dataType: matrix      description: state transfer probability matrix
              observation_matrix      dataType: matrix      description: observation probability matrix
              initial_probability     dataType: ndarray     description: initial state probability
      Output: self             dataType: obj       description: the trained model
    '''
    def train(self, transfer_matrix, observation_matrix, initial_probability):
