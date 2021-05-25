"""
@Filename:       Perceptron.py
@Author:         Ryuk
@Create Date:    2019-04-30
@Update Date:    2019-05-03
@Description:    Implement of perceptron.py
"""

import numpy as np
import preProcess
import pickle
import random

class  PerceptronClassifier:
    def __init__(self, norm_type="Normalization", iterations=500, learning_rate=0.01):
        self.norm_type = norm_type
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.gradients = None
        self.loss = None
        self.w = None
        self.b = None
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
       Function:  initializeParameter
       Description: initialize parameter
       Input:  feature_dim  dataType: int    description: feature dimension
       '''
    def initializeParameter(self, feature_dim):
        w = np.random.normal(0, 1, [feature_dim, 1])
        b = 0
        self.w = w
        self.b = b

    '''
       Function:  BackPropagate
       Description: BackPropagate function
       Input:  w            dataType: dict       description: the weights in network
               b            dataType: dict       description: the bias in network
               train_data   dataType: ndarray    description: train data
               train_label  dataType: ndarray    description: train label
       Output: gradients    dataType: dict       description: gradients
               cost         dataType: float      description: loss
       '''
    def backPropagate(self, train_data, train_label):
        num = train_label.shape[0]

        # forward
        A = self.sigmoid(np.dot(train_data, self.w) + self.b)
        cost = -1 / num * np.sum(train_label * np.log(A) + (1 - train_label) * np.log(1 - A))

        # backward
        dw = 1 / num * np.dot(train_data.T, A - train_label)
        db = 1 / num * np.sum(A - train_label)

        # save gradients
        gradients = {"dw": dw,
                     "db": db}
        return gradients, cost

    '''
          Function:  train
          Description: train the model
          Input:  train_data       dataType: ndarray   description: features
                  train_label      dataType: ndarray   description: labels
          Output: self             dataType: obj       description: the trained model
          '''
    def train(self, train_data, train_label):
        if self.norm_type == "Standardization":
            train_data = preProcess.Standardization(train_data)
        else:
            train_data = preProcess.Normalization(train_data)

        feature_dim = len(train_data[1])
        train_label = np.expand_dims(train_label, axis=1)
        self.initializeParameter(feature_dim)

        self.loss = []
        # training process
        for i in range(self.iterations):
            gradients, cost = self.backPropagate(train_data, train_label)
            # get the derivative
            dw = gradients["dw"]
            db = gradients["db"]

            # update parameter
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
            self.loss.append(cost)

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
            probability[i] = self.sigmoid(np.dot(self.w.T, test_data[i, :]) + self.b) # prediction = self.sigmoid(np.dot(self.w.T, test_data) + self.b) can speed up
            if probability[i] > 0:
                prediction[i] = 1
            else:
                prediction[i] = -1

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
        model = {'w': self.w, 'b': self.b}
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
        self.w = model['w']
        self.b = model['b']
        return self
