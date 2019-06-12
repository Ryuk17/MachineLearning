"""
@ Filename:       HMM.py
@ Author:         Danc1elion
@ Create Date:    2019-06-06   
@ Update Date:    2019-06-12
@ Description:    Implement HMM
"""
import numpy as np
import preProcess
import pickle

class HiddenMarkovModel:
    def __init__(self, Q, V, iterations=100):
        self.Q = Q                              # the set of states
        self.V = V                              # the set of observation
        self.N = len(Q)                         # length of Q
        self.M = len(V)                         # length of V
        self.A = None                           # transfer probability matrix
        self.B = None                           # observation probability matrix
        self.Pi = None                          # initial state probability
        self.P = None                           # probability  P(O|lambda)
        self.alpha = None                       # forward_probability
        self.beta = None                        # backward_probability
        self.iterations = iterations             # condition of convergence in EM

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
    def unsupervisedTrain(self, O):
        # initialize A, B, Pi
        self.A = np.zeros([self.N, self.N])/self.N
        self.B = np.zeros([self.N, self.M])/self.N
        self.Pi = np.random.random([self.N, 1])/self.N

        T = len(O)
        for it in range(self.iterations):
            # E step
            # get forward  and backward probability
            self.alpha, self.beta, self.P = self.calculateObservationProbability(O)

            post_state = np.multiply(self.alpha, self.beta)
            xi = np.zeros([self.N, self.N])
            for i in range(T):
                xi += (1 / self.P[i]) * np.outer(self.alpha[i - 1],self.beta[i]*self.B(O[i])) * self.A

            # M step
            self.Pi = post_state[0]/np.sum(post_state)
            for k in range(self.N):
                self.A[k] = xi[k]/np.sum(xi[k])

            gamma = np.zeros([self.N, self.M])


            for j in range(self.N):
                for k in range(self.M):
                    for t in range(T):
                        if O[t] == self.V[k]:
                            gamma[j][k] += 1

            self.B = gamma/post_state
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

    '''
      Function:  predict
      Description: predict
      Input:  test_data           dataType: list      description: test sequence
              method              dataType: string    description: "Viterbi" or "Approximate"
      Output: state_sequqnce      dataType: obj       description: best state sequqnce
    '''
    def predict(self, test_data, method="Viterbi"):
        sample_num = len(test_data)

        result = []
        if method == "Viterbi":
            for i in range(sample_num):
                result.append(self.Viterbi(test_data[i]))
            return result
        elif method == "Approximate":
            for i in range(sample_num):
                result.append(self.Approximate(test_data[i]))
            return result
        else:
            raise NameError('Unrecognized method')

    '''
      Function:  Viterbi
      Description: predict with Viterbi algorithm
      Input:  O                 dataType: list      description: test sequence
      Output: state_sequqnce    dataType: obj       description: best state sequqnce
    '''
    def Viterbi(self, O):
        T = len(O)
        delta = np.zeros([T, self.N])
        fai = np.zeros([T, self.N])
        state_sequence = np.zeros(T)
        delta[0] = np.multiply(self.Pi, self.B[:, O[0]])

        for t in range(1, T):
            delta_temp = np.tile(delta[t - 1], (self.N, 1)).T
            delta_temp = delta_temp * self.A
            delta[t] = np.multiply(np.max(delta_temp, axis=0), self.B[:, O[t]])
            fai[t] = np.argmax(delta_temp, axis=0)

        # end
        P = max(delta[T - 1])
        state_sequence[T - 1] = np.argmax(delta[T - 1])

        # recall
        t = T - 2
        while t >= 0:
            state_sequence[t] = fai[t + 1][int(state_sequence[t + 1])]
            t = t - 1
        return state_sequence

    '''
      Function:  Approximate
      Description: predict with Approximate algorithm
      Input:  O                 dataType: list      description: test sequence
      Output: state_sequqnce    dataType: obj       description: best state sequqnce
    '''
    def Approximate(self, O):
        T = len(O)
        state_sequence = np.zeros(T)
        alpha, beta, p = self.calculateObservationProbability(O)
        for t in range(T):
            gamma = np.multiply(alpha[t], beta[t])
            gamma = gamma/np.sum(gamma)
            state_sequence[t] = self.Q[np.argmax(gamma)]
        return state_sequence

    '''
         Function:  save
         Description: save the model as pkl
         Input:  filename    dataType: str   description: the path to save model
         '''
    def save(self, filename):
        f = open(filename, 'w')
        model = {'A': self.A, 'B': self.B, 'Pi': self.Pi}
        pickle.dump(model, f)
        f.close()

    '''
    Function:  load
    Description: load the model 
    Input:  filename    dataType: str   description: the path to save model
    Output: self        dataType: obj   description: the trained model
    '''
    def load(self, filename):
        f = open(filename)
        model = pickle.load(f)
        self.A = model['A']
        self.B = model['B']
        self.Pi = model['Pi']
        return self
