"""
@ Filename:       HMM.py
@ Author:         Ryuk
@ Create Date:    2019-06-06   
@ Update Date:    2019-06-16
@ Description:    Implement HMM
"""
import numpy as np
import pickle

class HiddenMarkovModel:
    def __init__(self, Q, V, iterations=10):
        self.Q = Q                              # the set of states
        self.V = V                              # the set of observation
        self.N = len(Q)                         # length of Q
        self.M = len(V)                         # length of V
        self.A = None                           # transfer probability matrix
        self.B = None                           # observation probability matrix
        self.Pi = None                          # initial state probability
        self.P = None                           # probability  P(O|lambda)
        self.iterations = iterations             # condition of convergence in EM

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
        for i in range(S):
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
        # there are more than one samples
        if len(state_sequence.shape) != 1:
            S = len(state_sequence)
            A = np.zeros([self.N, self.N])
            B = np.zeros([self.N, self.M])
            for i in range(S):
                a, b = self.parameterEstimation(state_sequence[i], observation_sequence[i])
                A += a
                B += b

            # calculate the initial probability
            initial_state = np.zeros([self.N])
            for i in range(S):
                for j in range(self.N):
                    if state_sequence[i][0] == self.Q[j]:
                        initial_state[j] += 1
            Pi = initial_state / S

            # transfer probability matrix
            for k in range(self.N):
                A[k, :] /= np.sum(A[k, :])

            # observation probability matrix
            for k in range(self.N):
                B[k, :] /= np.sum(B[k, :])

            self.A = A
            self.B = B
            self.Pi = Pi
            return self

        # there is only one samples
        else:
            A, B = self.parameterEstimation(state_sequence, observation_sequence)
            Pi = np.zeros([self.N, 1])
            Pi[state_sequence[0]] = 1

            # transfer probability matrix
            for k in range(self.N):
                A[k, :] /= np.sum(A[k, :])

            # observation probability matrix
            for k in range(self.N):
                B[k, :] /= np.sum(B[k, :])

            self.A = A
            self.B = B
            self.Pi = Pi
            return self

    '''
        Function:  calculateObservationProbability
        Description: calculate observation sequence probability  P(O|lambda)
        Input:    O               dataType: ndarray    description: observation sequences
                  A               dataType: ndarray    description: transfer probability matrix
                  B               dataType: ndarray    description: observation probability matrix
                  Pi              dataType: ndarray    description: initial state probability
        Output:   forward_prob    dataType: list       description: forward  probability at each time step
                  backward_prob   dataType: list       description: backward  probability at each time step
                  p               dataType: float      description: observation sequence probability  P(O|lambda)
      '''
    def calculateObservationProbability(self, O, A, B, Pi):
        T = len(O)
        N = len(self.Q)

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
            backward_probability.append(state.reshape(N, 1))
            t = t - 1
        # final result
        state = np.multiply(state, np.multiply(Pi, B[:, O[0]]))
        p = np.sum(state)
        return forward_probability, backward_probability, p

    '''
        Function:  EStep
        Description: estimate the parameters
        Input:    O               dataType: ndarray    description: observation sequences
                  A               dataType: ndarray    description: transfer probability matrix
                  B               dataType: ndarray    description: observation probability matrix
                  Pi              dataType: ndarray    description: initial state probability
        Output:   gamma           dataType: ndarray    description: single state gamma
                  xi              dataType: ndarray    description: double state xi
                  p               dataType: float      description: observation sequence probability  P(O|lambda)
    '''
    def EStep(self, O, A, B, Pi):
        T = len(O)
        alpha, beta, P = self.calculateObservationProbability(O, A, B, Pi)

        # the probability of being in state  i at time t given the observed sequence Y and the parameters
        gamma = np.zeros([T, self.N])
        for t in range(T):
            gamma[t, :] = np.multiply(alpha[t], beta[t]).reshape(self.N)
            gamma[t, :] /= np.sum(gamma[t, :])

        # the probability of being in state i and j at times t and t+1 respectively given the observed sequence Y and parameters
        xi = np.zeros((T - 1, self.N, self.N))
        for t in range(T - 1):
            denominator = 0.0
            for i in range(self.N):
                for j in range(self.N):
                    thing = 1.0
                    thing *= alpha[t][i]
                    thing *= A[i][j]
                    thing *= B[j][t + 1]
                    thing *= beta[t + 1][j]
                    denominator += thing
            for i in range(self.N):
                for j in range(self.N):
                    numerator = 1.0
                    numerator *= alpha[t][i]
                    numerator *= A[i][j]
                    numerator *= B[j][t + 1]
                    numerator *= beta[t + 1][j]
                    xi[t][i][j] = numerator / denominator

        return gamma, xi, P

    '''
        Function:  MStep
        Description: maximize the parameters
        Input:    O               dataType: ndarray    description: observation sequences
                  A               dataType: ndarray    description: transfer probability matrix
                  B               dataType: ndarray    description: observation probability matrix
                  Pi              dataType: ndarray    description: initial state probability
                  gamma           dataType: ndarray    description: single state gamma
                  xi              dataType: ndarray    description: double state xi
        Output:   A               dataType: ndarray    description: new transfer probability matrix
                  B               dataType: ndarray    description: new observation probability matrix
                  Pi              dataType: ndarray    description: new initial state probability
    '''
    def MStep(self, O, gamma, xi):
        T = len(O)
        # update A
        A = np.sum(xi[:T - 1], axis=0) / np.sum(gamma[:T - 1], axis=0)
        # update Pi
        Pi = gamma[0]
        # update B
        temp = np.zeros([self.N, self.M])
        for t in range(T):
            for j in range(self.N):
                for k in range(self.M):
                    if O[t] == self.V[k]:
                        flag = 1
                    else:
                        flag = 0
                    temp[j][k] = gamma[t][j] * flag

        B = temp / gamma.sum()
        return A, B, Pi

    '''
      Function: initializeParameters
      Description: initialize A, B, Pi
      Output:     A               dataType: ndarray    description: initial transfer probability matrix
                  B               dataType: ndarray    description: initial observation probability matrix
                  Pi              dataType: ndarray    description: initial initial state probability
    '''
    def initializeParameters(self):
        # initialize A, B, Pi, which follow the sum of each row is equal to 1
        A = np.random.dirichlet(np.ones(self.N), size=self.N)
        B = np.random.dirichlet(np.ones(self.M), size=self.N)
        Pi = np.random.dirichlet(np.ones(self.N), size=1)
        return A, B, Pi

    '''
      Function:  unsupervisedTrain
      Description: train the model with unsupervised algorithm
      Input:  observation_sequence     dataType: list      description: observation sequence
      Output: self                     dataType: obj       description: the trained model
    '''
    def unsupervisedTrain(self, O, epsilon=0.0001):
        # there are more than one samples
        if len(O.shape) != 1:
            sample_num = len(O)
            A, B, Pi = self.initializeParameters()

            for n in range(self.iterations):
                denominator_a = 0.0
                numerator_a = 0.0
                denominator_b = 0.0
                numerator_b = 0.0
                denominator_pi = 0.0
                numerator_pi = 0.0

                for i in range(sample_num):
                    T = len(O[i])
                    # E step
                    gamma, xi, P = self.EStep(O[i], A, B, Pi)

                    # M step
                    # parameters for A
                    numerator_a += 1/P * np.sum(xi, axis=0)
                    denominator_a += 1/P * np.sum(gamma[0:T-1], axis=0)

                    # parameters for B
                    temp = np.zeros([self.N, self.M])
                    for t in range(T):
                        for j in range(self.N):
                            for k in range(self.M):
                                if O[i][t] == self.V[k]:
                                    flag = 1
                                else:
                                    flag = 0
                                temp[j][k] += gamma[t][j] * flag

                    numerator_b += 1/P * temp
                    denominator_b += 1/P * np.sum(gamma, axis=0)

                    # parameters for Pi
                    numerator_pi += 1/P * gamma[0]
                    denominator_pi += 1/P

                # update A
                A = numerator_a / denominator_a

                # update B
                for i in range(self.N):
                    B[i, :] = numerator_b[i, :] / denominator_b[i]

                # update Pi
                Pi = numerator_pi / denominator_pi

            self.A = A
            self.B = B
            self.Pi = Pi
            return self

        # there is only one samples
        else:
            # initialize A, B, Pi
            A = np.ones([self.N, self.N])/self.N
            B = np.ones([self.N, self.M])/self.N
            Pi = np.random.random([self.N])/self.N

            # EM algorithm
            for it in range(self.iterations):
                gamma, xi, old_prob = self.EStep(O, A, B, Pi)
                A, B, Pi = self.MStep(O, gamma, xi)
                _, _, new_prob = self.calculateObservationProbability(O, A, B, Pi)

                if abs(new_prob -old_prob) < epsilon:
                    break

            self.A = A
            self.B = B
            self.Pi = Pi
            return self

    '''
      Function:  train
      Description: train the model
      Input:  state_sequence           dataType: list      description: state sequence
              observation_sequence     dataType: list      description: observation sequence
      Output: self                     dataType: obj       description: the trained model
    '''
    def train(self, observation_sequence, state_sequence=None):
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
            if len(test_data.shape) != 1:
                for i in range(sample_num):
                    result.append(self.Viterbi(test_data[i]))
                return result
            else:
                return self.Viterbi(test_data)
        elif method == "Approximate":
            if len(test_data.shape) != 1:
                for i in range(sample_num):
                    result.append(self.Approximate(test_data[i]))
                return result
            else:
                return self.Approximate(test_data)
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
        alpha, beta, p = self.calculateObservationProbability(O, self.A, self.B, self.Pi)
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
