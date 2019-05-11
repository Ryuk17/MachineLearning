"""
@ Filename:       Regression.py
@ Author:         Danc1elion
@ Create Date:    2019-05-05
@ Update Date:    2019-05-06
@ Description:    Implement linear regression
"""
import numpy as np
import preProcess
import pickle
import random
import matplotlib.pyplot as plt

class RegressionAnalysis:
    def __init__(self, norm_type="Normalization",regression_type="Standard", k=1.0, lamda=0.2, learning_rate=0.01, iterations=100):
        self.norm_type = norm_type
        self.regression_type = regression_type
        self.k = k                                  # parameter for local weight linear regression
        self.lamda = lamda                          # parameter for ridge regression
        self.learning_rate = learning_rate          # parameter for forward step regression
        self.iterations = iterations                # parameter for forward step regression
        self.w = None
        self.parameters = None
        self.prediction = None
        self.probability = None

    '''
       Function:  standardLinearRegression
       Description: standard Linear Regression, w =(X.T*X)-1*X.T*y
       Input:  x          dataType: ndarray      description: x 
               y          dataType: ndarray      description: y
       Output: w          dataType: ndarray      description: weights
       '''
    def standardLinearRegression(self, x, y):
        if self.norm_type == "Standardization":
            x = preProcess.Standardization(x)
        else:
            x = preProcess.Normalization(x)

        xTx = np.dot(x.T, x)
        if np.linalg.det(xTx) == 0:   # calculate the Determinant of xTx
            print("Error: Singluar Matrix !")
            return
        w = np.dot(np.linalg.inv(xTx), np.dot(x.T, y))
        return w

    '''
       Function:  LWLinearRegression
       Description: locally weighted linear regression, w = (X.T*W*X)-1*X.T*W*y
       Input:  x          dataType: ndarray      description: x 
               y          dataType: ndarray      description: y
       Output: w          dataType: ndarray      description: weights
       '''
    def LWLinearRegression(self, x, y, sample):
        if self.norm_type == "Standardization":
            x = preProcess.Standardization(x)
        else:
            x = preProcess.Normalization(x)

        sample_num = len(x)
        weights = np.eye(sample_num)
        for i in range(sample_num):
            diff = sample - x[i, :]
            weights[i, i] = np.exp(np.dot(diff, diff.T)/(-2 * self.k ** 2))
        xTx = np.dot(x.T, np.dot(weights, x))
        if np.linalg.det(xTx) == 0:
            print("Error: Singluar Matrix !")
            return
        result = np.dot(np.linalg.inv(xTx), np.dot(x.T, np.dot(weights, y)))
        return result

    '''
       Function:  ridgeRegression
       Description: ridge linear regression, w = (X.T*X+ LAMDA I)-1*X.T*y
       Input:  x          dataType: ndarray      description: x 
               y          dataType: ndarray      description: y
       Output: w          dataType: ndarray      description: weights
       '''
    def ridgeRegression(self, x, y):
        if self.norm_type == "Standardization":
            x = preProcess.Standardization(x)
        else:
            x = preProcess.Normalization(x)

        feature_dim = len(x[0])
        xTx = np.dot(x.T, x)
        matrix = xTx + np.exp(feature_dim)*self.lamda
        if np.linalg.det(xTx) == 0:
            print("Error: Singluar Matrix !")
            return
        w = np.dot(np.linalg.inv(matrix), np.dot(x.T, y))
        return w

    '''
       Function:  lasso Regression
       Description: lasso linear regression, 
       Input:  x          dataType: ndarray      description: x 
               y          dataType: ndarray      description: y
       Output: w          dataType: ndarray      description: weights
       '''
    def lassoRegression(self, x, y):
        if self.norm_type == "Standardization":
            x = preProcess.Standardization(x)
        else:
            x = preProcess.Normalization(x)

        sample_num, feataure_dim = np.shape(x)
        w = np.zeros([feataure_dim, 1])
        for i in range(self.iterations):
            last_w = w
            w[i] = np.dot(x[i, :], (y[i] - x[i, :] * last_w.T))/np.dot(x[i, :], x[i, :].T)
        return w


    '''
       Function:  forwardstep Regression
       Description: forward step linear regression, 
       Input:  x          dataType: ndarray      description: x 
               y          dataType: ndarray      description: y
       Output: w          dataType: ndarray      description: weights
       '''
    def forwardstepRegression(self, x, y):
        if self.norm_type == "Standardization":
            x = preProcess.Standardization(x)
        else:
            x = preProcess.Normalization(x)

        sample_num, feature_dim = np.shape(x)
        w = np.zeros([self.iterations, feature_dim])
        best_w = np.zeros([feature_dim, 1])
        for i in range(self.iterations):
            min_error = np.inf
            for j in range(feature_dim):
                for sign in [-1, 1]:
                    temp_w = best_w
                    temp_w[j] += sign * self.learning_rate
                    y_hat = np.dot(x, temp_w)
                    error = ((y - y_hat) ** 2).sum()                # MSE
                    if error < min_error:                           # save the best parameters
                        min_error = error
                        best_w = temp_w
            w[i, :] = best_w.T
        return w

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

        if self.regression_type == "Standard":
            self.w = self.standardLinearRegression(train_data, train_label)
        elif self.regression_type == "Localweight":
            self.w = self.LWLinearRegression(train_data, train_label)
        elif self.regression_type == "Ridge":
            self.w = self.ridgeRegression(train_data, train_label)
        elif self.regression_type == "Lasso":
            self.w = self.lassoRegression(train_data, train_label)
        elif self.regression_type == "Forwardstep":
            self.w = self.forwardstepRegression(train_data, train_label)
        else:
            print("Error Regression Type!")
        return self

    '''
       Function:  predict
       Description: predict the testing set 
       Input:  test_data       dataType: ndarray   description: features
               prob             dataType: bool      description: return probaility of label
       Output: prediction       dataType: ndarray   description: the prediction results for testing set
       '''
    def predict(self, x, prob="False"):
        # Normalization
        if self.norm_type == "Standardization":
            x = preProcess.Standardization(x)
        else:
            x = preProcess.Normalization(x)

        y = np.dot(x, self.w)
        self.prediction = y
        return y

    '''
    Function:  plot
    Description: show regression result
    Input:  test_label dataType: ndarray   description: labels of test data
    Output: accuracy   dataType: float     description: detection accuarcy
    '''
    def plot(self, test_label):
        # test_label = np.expand_dims(test_label, axis=1)
        prediction = self.prediction
        plot1 = plt.plot(test_label, 'r*', label='Regression values')
        plot2 = plt.plot(prediction, 'b', label='Real values')
        plt.xlabel('X ')
        plt.ylabel('Y')
        plt.legend(loc=3)
        plt.title('Regression')
        plt.show()

    '''
          Function:  save
          Description: save the model as pkl
          Input:  filename    dataType: str   description: the path to save model
          '''

    def save(self, filename):
        f = open(filename, 'w')
        pickle.dump(self.w, f)
        f.close()

    '''
    Function:  load
    Description: load the model 
    Input:  filename    dataType: str   description: the path to save model
    Output: self        dataType: obj   description: the trained model
    '''

    def load(self, filename):
        f = open(filename)
        self.w = pickle.load(f)
        return self

