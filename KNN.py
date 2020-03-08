"""
@Filename:       KNN.py
@Author:         Ryuk
@Create Date:    2019-04-29
@Update Date:    2019-05-03
@Description:    Implement of KNN
"""

import numpy as np
import operator as op

class KNNClassifier:
    def __init__(self, k, norm_type="Normalization"):
        self.k = k
        self.norm_type = "Normalization"
        self.x_train = None
        self.y_train = None

    '''
    Function:  Normalization
    Description: Normalize input data. For vector x, the normalization process is given by
                 normalization(x) = (x - min(x))/(max(x) - min(x))
    Input:  data        dataType: ndarray   description: input data
    Output: norm_data   dataType: ndarray   description: output data after normalization
    '''
    def Normalization(self, data):
        # get the max and min value of each column
        min_value = data.min(axis=0)
        max_value = data.max(axis=0)
        diff = max_value - min_value
        # normalization
        min_data = np.tile(min_value, (data.shape[0], 1))
        norm_data = (data - min_data)/np.tile(diff, (data.shape[0], 1))
        return norm_data

    '''
    Function:  Standardization
    Description: Standardize input data. For vector x, the normalization process is given by
                 Standardization(x) = x - mean(x)/std(x)
    Input:  data            dataType: ndarray   description: input data
    Output: standard_data   dataType: ndarray   description: output data after standardization
    '''
    def Standardization(self, data):
        # get the mean and the variance of each column
        mean_value = data.mean(axis=0)
        var_value = data.std(axis=0)
        standard_data = (data - np.tile(mean_value, (data.shape[0], 1)))/np.tile(var_value, (data.shape[0], 1))
        return standard_data

    '''
    Function:  train
    Description: train the model
    Input:  train_data       dataType: ndarray   description: features
            test_data        dataType: ndarray   description: labels
    Output: self             dataType: obj       description: 
    '''
    def train(self, train_data, train_label):
        if self.normType == "Standardization":
            train_data = self.Standardization(train_data)
        else:
            train_data = self.Normalization(train_data)
        self.x_train = train_data
        self.y_train = train_label
        return self

    '''
    Function:  predict
    Description: give the prediction for test data
    Input:  test_data    dataType: ndarray   description: data for testing
            test_abel    dataType: ndarray   description: labels of train data
            norm_type    dataType: string    description: type of normalization, default:Normalization
            probability  dataType: bool      description: if true return label and probability, else return label only
            showResult   dataType: bool      description: display the prediction result
    Output: results      dataType: ndarray   description: label or probability
    '''
    def predict(self, test_data):
        # Normalization
        if self.normType == "Standardization":
            testData = self.Standardization(test_data)
        else:
            testData = self.Normalization(test_data)

        test_num = testData.shape[0]
        prediction = np.zeros([test_num, 1])
        probability = np.zeros([test_num, 1])
        # predict each samples in test data
        for i in range(test_num):
            prediction[i], probability[i] = self.calcuateDistance(testData[i], self.x_train, self.y_train, self.k)

        return prediction

    '''
    Function:  calcuateDistance
    Description: calcuate the distance between input vector and train data
    Input:  input       dataType: ndarray   description: input vector
            traind_ata  dataType: ndarray   description: data for training
            train_label dataType: ndarray   description: labels of train data
            k           dataType: int       description: select the first k distances
    Output: prob        dataType: float     description: max probability of prediction 
            label       dataType: int       description: prediction label of input vector
    '''
    def calcuateDistance(self, input, train_data, train_label, k):
        train_num = train_data.shape[0]
        # calcuate the distances
        distances = np.tile(input, (train_num, 1)) - train_data
        distances = distances**2
        distances = distances.sum(axis=1)
        distances = distances**0.5

        # get the labels of the first k distances
        disIndex = distances.argsort()
        labelCount = {}
        for i in range(k):
            label = train_label[disIndex[i]]
            labelCount[label] = labelCount.get(label, 0) + 1

        prediction = sorted(labelCount.items(), key=op.itemgetter(1), reverse=True)
        label = prediction[0][0]
        prob = prediction[0][1]/k
        return label, prob

    '''
    Function:  showDetectionResult
    Description: show detection result
    Input:  test_data  dataType: ndarray   description: data for test
            test_label dataType: ndarray   description: labels of test data
    Output: accuracy   dataType: float     description: detection accuarcy
    '''
    def showDetectionResult(self, test_data, test_label):
        test_label = np.expand_dims(test_label, axis=1)
        prediction = self.predict(test_data)
        accuarcy = sum(prediction == test_label)/len(test_label)
        return accuarcy
