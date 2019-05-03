"""
@Filename:       LogisticRegression.py
@Author:         Danc1elion
@Date:           2019-04-30
@Update Date:    2019-05-03
@Description:    Implement of logistic regression
"""

import numpy as np
import preProcess
import pickle
import random


class LogisticRegressionClassifier:
    def __init__(self,norm_type="Normalization"):
        self.norm_type = norm_type
        self.weights = None
        self.prediction = None
        self.probability = None
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
       Function:  updataAlpha
       Description: updata Alpha in each sample
       Input: alpha       dataType: float     description: original alpha
              method      dataTpye: int       description: update method of alpha
       Output: output     dataType: float     description: output
       '''
    def updataAlpha(self, alpha, epoch, method=1):
        if method == 1:
            alpha = 0.95 ** epoch * alpha
        elif method == 2:
            k = 3
            alpha = k/(epoch ** 0.5) * alpha
        elif method == 3:
            decay_rate = 0.001
            alpha = alpha / (1 + decay_rate * epoch)
        return alpha

    '''
       Function:  train
       Description: train the model
       Input:  train_data       dataType: ndarray   description: features
               train_label      dataType: ndarray   description: labels
               method           dataType: string    description: "GA":Gradient Ascent; "SGA": Stochastic Gradient Ascent
               alpha            dataType: float     description: the stride of the target
               iterations       dataType: int       description: the times of iteration
       Output: self             dataType: obj       description: the trained model
       '''
    def train(self, train_data, train_label, method="GA", alpha=0.1, iterations=100):
        if self.norm_type == "Standardization":
            train_data = preProcess.Standardization(train_data)
        else:
            train_data = preProcess.Normalization(train_data)

        train_label = np.expand_dims(train_label, axis=1)
        feature_dim = len(train_data[1])


        if method == "GA":
            weights = np.random.normal(0, 1, [feature_dim, 1])
            for i in range(iterations):
                pred = self.sigmoid(np.dot(train_data, weights))
                errors = train_label - pred
                # update the weights
                weights = weights + alpha * np.dot(train_data.T, errors)
            self.weights = weights
            return self

        if method == "SGA":
            weights = np.random.normal(0, 1, feature_dim)
            sample_num = len(train_data)
            random_index = np.random.randint(sample_num, size=sample_num)
            for i in range(iterations):
                for j in range(sample_num):
                    alpha = self.updataAlpha(alpha, i, 1)
                    pred = self.sigmoid(np.dot(train_data[random_index[j], :], weights))
                    sample_error = train_label[random_index[j]] - pred
                    weights = weights + alpha * sample_error * train_data[random_index[j], :]

            self.weights = weights
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
            probability[i] = self.sigmoid(np.dot(test_data[i, :], self.weights))
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
