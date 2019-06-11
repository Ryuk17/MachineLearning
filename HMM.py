"""
@ Filename:       HMM.py
@ Author:         Danc1elion
@ Create Date:    2019-06-06   
@ Update Date:    2019-06-11
@ Description:    Implement HMM
"""
import numpy as np
import preProcess

class HiddenMarkovModel:
    def __init__(self, iterations=100):
        self.Q = None                           # the set of states
        self.V = None                           # the set of observation
        self.N = None                           # length of Q
        self.M = None                           # length of V
        self.A = None                           # transfer probability matrix
        self.B = None                           # observation probability matrix
        self.Pi = None                          # initial state probability
        self.P = None                           # probability  P(O|lambda)
        self.alpha = None                       # forward_probability
        self.beta = None                        # backward_probability
        self.iterations = iterations              # condition of convergence in EM

    '''
      Function:  calculateObservationProbability
      Description: calculate observation sequence probability  P(O|lambda)
      Input:    O               dataType: list       description: observation sequence
      Output:   forward_prob    dataType: list       description: forward  probability at each time step
                backward_prob   dataType: list       description: backward  probability at each time step
                p               dataType: float      description: observation sequence probability  P(O|lambda)
    '''
    def calculateObservationProbability(self, O):
        T = len(O)
        N = len(self.Q)
        # forward algorithm
        forward_probability = []
        # calculate the initial state
        state = np.multiply(self.Pi, self.B[:, O[0]]).reshape(N, 1)
        forward_probability.append(state)
        # recursion
        t = 1
        while t < T:
            state = np.multiply(state, self.A)
            state = np.sum(state, axis=0)
            state = np.multiply(state, self.B[:, O[t]]).reshape(N, 1)
            forward_probability = []
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
            temp = np.multiply(self.B[:, O[t]], self.A)
            temp = np.multiply(state, temp)
            state = np.sum(temp, axis=1)
            backward_probability.append(state)
            t = t - 1
        # final result
        state = np.multiply(state, np.multiply(self.Pi, self.B[:, O[0]]))
        p = np.sum(state)

        self.alpha = forward_probability
        self.beta = backward_probability
        return forward_probability, backward_probability, p

    '''
      Function:  getGamma
      Description: calculate the probability of state q at time t. In Ref[4] P179 Eq.(10.24)
      Input:  t     dataType: int     description: time t
      Output: r     dataType: float   description: the probability of state q at time t
    '''
    def getGamma(self, t):
        r = self.alpha[t] * self.beta[t]
        r = r/r.sum()
        return r

    '''
      Function:  getXi
      Description: calculate the probability of state i at time t and state j and time t+1. In Ref[4] P179 Eq.(10.25)
      Input:  t     dataType: int     description: time t
      Output: r     dataType: float   description: the probability of state q at time t
    '''
    def getXi(self, t, O):
        r1 = np.dot(self.alpha[t], self.A)
        r2 = np.dot(self.B[:, O[t+1]], self.beta[t])
        return

    '''
      Function: parameterEstimation
      Description: estimate transfer probability matrix and observation probability matrix
      Input:  I           dataType: list      description: state sequence
              O           dataType: list      description: observation sequence
      Output: A           dataType: ndarray   description: transfer probability matrix
              B           dataType: ndarray   description: observation probability matrix
    '''
    def parameterEstimation(self, I, O):
        S = len(I)
        # calculate transfer probability matrix
        A = np.zeros([self.N, self.N])
        for i in range(S - 1):
            for m in range(self.N):
                for n in range(self.N):
                    if I[i] == self.Q[m] and I[i + 1] == self.Q[n]:  # (i, t)->(j, t+1)
                        A[m][n] += 1

        # calculate observation probability matrix
        B = np.zeros([self.N, self.M])
        for i in range(S - 1):
            for m in range(self.N):
                for n in range(self.M):
                    if I[i] == self.Q[m] and O[i] == self.V[n]:  # when state is m and observation is n
                        B[m][n] += 1
        return A, B

    '''
      Function:  supervisedTrain
      Description: train the model with supervised algorithm
      Input:  state_sequence           dataType: list      description: state sequence
              observation_sequence     dataType: list      description: observation sequence
      Output: self                     dataType: obj       description: the trained model
    '''
    def supervisedTrain(self, state_sequence, observation_sequence):
        S = len(state_sequence)
        A = np.zeros([self.N, self.N])
        B = np.zeros([self.N, self.M])
        initial_state = np.zeros([S, 1])

        for i in range(S):
            a, b = self.parameterEstimation(state_sequence[i], observation_sequence[i])
            A += a
            B += b
            initial_state[i] = state_sequence[i][0]

        # transfer probability matrix
        for k in range(self.N):
            A[k, :] /= np.sum(A[k, :])

        # observation probability matrix
        for k in range(self.N):
            B[k, :] /= np.sum(B[k, :])

        # calculate the initial probability
        Pi = np.zeros([self.N, 1])
        for i in range(S):
            for j in range(self.N):
                if initial_state[i] == self.Q[j]:
                    Pi[j] += 1
        Pi = Pi/S

        self.A = A
        self.B = B
        self.Pi = Pi
        return self

    '''
      Function:  unsupervisedTrain
      Description: train the model with unsupervised algorithm
      Input:  observation_sequence     dataType: list      description: observation sequence
      Output: self                     dataType: obj       description: the trained model
    '''
    def unsupervisedTrain(self, observation_sequence):
        # initialize A, B, Pi
        self.A = np.zeros([self.N, self.N])/self.N
        self.B = np.zeros([self.N, self.M])/self.N
        self.Pi = np.random.random([self.N, 1])/self.N

        T = len(observation_sequence)

        for it in range(self.iterations):
          


        return self

    '''
      Function:  train
      Description: train the model
      Input:  state_sequence           dataType: list      description: state sequence
              observation_sequence     dataType: list      description: observation sequence
      Output: self                     dataType: obj       description: the trained model
    '''
    def train(self, state_sequence, observation_sequence):

        if state_sequence is not None:
            return self.supervisedTrain(state_sequence, observation_sequence)
        else:
            return self.unsupervisedTrain(observation_sequence)

