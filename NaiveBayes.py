"""
@Filename:       NaiveBayes.py
@Author:         Ryuk
@Create Date:    2019-05-02
@Update Date:    2019-05-03
@Description:    Implement of naive Bayes
"""

import numpy as np
import operator as op
import preProcess
import math
import pickle


class BayesClassifier:
    def __init__(self, norm_type="Normalization", laplace=1):
        self.norm_type = norm_type
        self.laplace = laplace
        self.label_value = None
        self.feature_value = None
        self.S = None
        self.prior_probability = None
        self.conditional_probability = None
        self.prediction = None
        self.probability = None
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

        label_count = {}
        feature_dim = len(train_data[1])

        # get the number of each labels
        for c in train_label:
            label_count[c] = label_count.get(c, 0) + 1
        label_value = sorted(label_count.items(), key=op.itemgetter(0), reverse=False)
        self.label_value = label_value

        K = len(label_value)         # the number of unique labels
        N = len(train_label)         # the number of samples

        # get the prior probability
        prior_probability = {}
        for key in range(len(label_value)):
            prior_probability[label_value[key][0]] = (label_value[key][1] + self.laplace) / (N + K * self.laplace)  # laplace smooth
        self.prior_probability = prior_probability

        # get the value set of each feature
        feature_value = []  # feature with different value
        S = []  # the number of unique values of each feature
        for feat in range(feature_dim):
            unique_feature = np.unique(train_data[:, feat])
            S.append(len(unique_feature))
            feature_value.append(unique_feature)
        self.S = S
        self.feature_value = feature_value

        # calculate the conditional probability
        prob = []
        # calculate the count (x = a & y = c)
        for j in range(feature_dim):
            count = np.zeros([S[j], len(label_count)])  # the range of label start with 1
            feature_temp = train_data[:, j]
            feature_value_temp = feature_value[j]
            for i in range(len(feature_temp)):
                for k in range(len(feature_value_temp)):
                    for t in range(len(label_count)):
                        if feature_temp[i] == feature_value_temp[k] and train_label[i] == label_value[t][0]:
                            count[k][t] += 1             # x = value and y = label
            # calculate the conditional probability
            for m in range(len(label_value)):
                count[:, m] = (count[:, m] + self.laplace) / (label_value[m][1] + self.laplace*S[j])  # laplace smoothing
            # print(count)
            prob.append(count)
        self.conditional_probability = prob
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
            result = self.classify(test_data[i, :])
            result = sorted(result.items(), key=op.itemgetter(1), reverse=True)
            prediction[i] = result[0][0]

        self.prediction = prediction
        self.probability = probability
        if prob:
            return probability
        else:
            return prediction
    '''
        Function:  classify
        Description: predict the testing set 
        Input:  sample      dataType: ndarray     description: input vector to be classified
        Output: label       dataType: ndarray     description: the prediction results of input
     '''
    def classify(self, sample):
        predict = {}
        for m in range(len(self.label_value)):
            temp = self.prior_probability[self.label_value[m][0]]  # get the prior_probability of m-th label in label_value
            for n in range(len(sample)):
                if sample[n] in self.feature_value[n]:
                    # print(m, n)
                    index = np.where(self.feature_value[n] == sample[n])[0][0]
                    temp = temp * self.conditional_probability[n][index][m]
                else:
                    temp = self.laplace / (self.S[n] * self.laplace)  # if the value of feature is not in training set, return the laplace smoothing
            predict[self.label_value[m][0]] = temp
        return predict

    '''
    Function:  accuracy
    Description: show detection result
    Input:  test_data  dataType: ndarray   description: data for test
            test_label dataType: ndarray   description: labels of test data
    Output: accuracy   dataType: float     description: detection accuarcy
    '''
    def accuarcy(self, test_label):
        test_label = np.expand_dims(test_label, axis=1)
        prediction = self.prediction
        accuarcy = sum(prediction == test_label)/len(test_label)
        return accuarcy






