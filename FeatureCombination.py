"""
@ Filename:      FeatureCombination.py
@ Author:         Ryuk
@ Create Date:    2019-11-18
@ Update Date:    2019-11-20
@ Description:    Implement FM
"""

import numpy as np
import preProcess
import pickle

class FM:
    def __init__(self, n, norm_type="Standardization", k=5):
        self.norm_type = norm_type
        self.n = n                                      # the number of feature
        self.k = k                                      # the dimension of latency
        self.w_0 = 0                                    # numerical parameter
        self.W = np.random.random([self.n, 1])          # one order parameter
        self.V = np.random.random([self.n, self.k])     # second order parameter
        self.sample_num = None                          # the number of samples of trainset

    '''
       Function:  sigmoid
       Description: sigmoid function
       Input:  x          dataType: ndarray   description: input vector
               derivative dataType: bool      description: whether to calculate the derivative of sigmoid
       Output: output     dataType: float     description: output
       '''
    def sigmoid(self, x, derivative=False):
        output = 1/(1 + np.exp(-x))
        if derivative:
            output = output * (1 - output)
        return output


    '''
       Function:  train
       Description: train the model
       Input:  train_data       dataType: ndarray   description: features
               train_label      dataType: ndarray   description: labels
               alpha            dataType: float     description: the stride of the target
               iterations       dataType: int       description: the times of iteration
       Output: self             dataType: obj       description: the trained model
       '''
    def train(self, train_data, train_label, alpha=0.01, iterations=100):
        if self.norm_type == "Standardization":
            train_data = preProcess.Standardization(train_data)
        else:
            train_data = preProcess.Normalization(train_data)

        for epoch in range(iterations):
            for id in range(self.sample_num):

                # second order computation
                inter_1 = train_data[id] * self.V
                inter_2 = np.multiply(train_data[id], train_data[id]) * np.multiply(self.V, self.V)
                interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.

                # prediction result
                pred = self.w_0 + train_data[id] * self.W + interaction

                # calculate loss, cross entropy
                base = [np.log(self.sigmoid(train_label[id] * float(pred))) - 1] * train_label

                # update numerical parameters
                self.w_0 -= alpha * base

                x = train_data[id]
                for i in range(self.n):
                    # update first-order parameter
                    if train_data[id, i] != 0:
                        self.W[id, i] -= alpha * base  * train_data[id, i]
                        for j in range(self.n):
                            # update second-order parameter
                            self.V[i, j] -= alpha * base * (
                                    train_data[id, i] * self.V[j, i] * train_data[id, j] - self.V[i, j] * train_data[id, i] * train_data[id, i])

        return self


    '''
       Function:  predict
       Description: predict the testing set 
       Input:  train_data       dataType: ndarray   description: features
               prob             dataType: bool      description: return probaility of label
       Output: prediction       dataType: ndarray   description: the prediction results for testing set
       '''
    def predict(self, test_data, prob="False"):
        # Normalization
        if self.norm_type == "Standardization":
            test_data = preProcess.Standardization(test_data)
        else:
            test_data = preProcess.Normalization(test_data)

        test_num = test_data.shape[0]
        prediction = np.zeros([test_num, 1])
        probability = np.zeros([test_num, 1])
        for i in range(test_num):

            inter_1 = test_data[i] * self.V
            inter_2 = np.multiply(test_data[i], test_data[i]) * np.multiply(self.V, self.V)
            interaction = sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
            pre = self.w_0 + test_data[i] * self.W + interaction
            probability = self.sigmoid(float(pre))

            if probability[i] > 0.5:
                prediction[i] = 1
            else:
                prediction[i] = 0.5

        self.prediction = prediction
        self.probability = probability
        if prob:
            return probability
        else:
            return prediction


    '''
    Function:  accuracy
    Description: show detection result
    Input:  test_label dataType: ndarray   description: labels of test data
    Output: accuracy   dataType: float     description: detection accuarcy
    '''
    def accuarcy(self, test_label):
        test_label = np.expand_dims(test_label, axis=1)
        prediction = self.prediction
        accuarcy = sum(prediction == test_label)/len(test_label)
        return accuarcy

    '''
       Function:  save
       Description: save the model as pkl
       Input:  filename    dataType: str   description: the path to save model
       '''
    def save(self, filename):
        f = open(filename, 'w')
        pickle.dump(self.weights, f)
        f.close()

    '''
    Function:  load
    Description: load the model 
    Input:  filename    dataType: str   description: the path to save model
    Output: self        dataType: obj   description: the trained model
    '''
    def load(self, filename):
        f = open(filename)
        self.weights = pickle.load(f)
        return self
