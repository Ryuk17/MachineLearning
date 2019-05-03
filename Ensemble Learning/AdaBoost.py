"""
@Filename:       AdaptiveBoost.py
@Author:         Danc1elion
@Create Date:    2019-05-03
@Update Date:    2019-05-03
@Description:    Implement of Adaptive Boosting
"""

import numpy as np
import preProcess
import pickle
import random
import SVM
import math

class Adaboost:
    def __init__(self, norm_type="Normalization", iterations=5, base_classifier="SVM"):
        self.iterations = iterations
        self.norm_type = norm_type
        self.base_classifier = SVM.SVMClassifier()
        self.prediction = None
        self.probability = None
        self.classifier_set = None

    '''
       Function:  baseClassifier
       Description: generate weak classifier
       Input: train_data            dataType: ndarray        description: train_data
              train_label           dataType: ndarray        description: train_label
              w                     dataType: ndarray        description: weight
       Output: clf                  dataType: object         description: weak classifier
               weighted_error       dataType: float          description: weighted error
               base_predictions     dataType: object         description: base predictions
                
       '''
    def baseClassifier(self, train_data, train_label, w):
        sample_num = len(train_data)
        error_index = np.ones([sample_num, 1])
        clf = self.base_classifier
        clf.train(train_data, train_label)
        base_predictions = np.sign(clf.predict(train_data))

        for i in range(sample_num):
            if base_predictions[i] == train_label[i]:
                error_index[i] = 0
        weighted_error = np.dot(w.T, error_index)
        return clf, weighted_error, base_predictions

    '''
        Function:  updataAlpha
        Description: updata alpha
        Input:  error            dataType: float     description: weighted error
        Output: new_alpha        dataType: float     description: new alpha
            '''
    def updateAlpha(self, error):
        temp = (1.0 - error)/max(error, 10e-6)
        new_alpha = 1/2 * math.log(temp, math.e)
        return new_alpha

    '''
        Function:  train
        Description: train the model
        Input:  train_data       dataType: ndarray   description: features
                train_label      dataType: ndarray   description: labels
        Output: clf_set          dataType: list      description: classifiers set
          '''
    def train(self, train_data, train_label):
        if self.norm_type == "Standardization":
            train_data = preProcess.Standardization(train_data)
        else:
            train_data = preProcess.Normalization(train_data)

        train_label = np.expand_dims(train_label, axis=1)
        sample_num = len(train_data)

        weak_classifier = []

        # initialize weights
        w = np.ones([sample_num, 1])
        w = w/sample_num

        # predictions
        agg_predicts = np.zeros([sample_num, 1]) # aggregate value of prediction

        # start train
        for i in range(self.iterations):
            base_clf, error, base_prediction = self.baseClassifier(train_data, train_label, w)
            alpha = self.updateAlpha(error)
            weak_classifier.append((alpha, base_clf))

            # update parameters in page of 139 Eq.(8.4)
            expon = np.multiply(-1 * alpha * train_label, base_prediction)
            w = np.multiply(w, np.exp(expon))
            w = w/w.sum()

            # calculate the total error rate
            agg_predicts += alpha*base_prediction
            error_rate = np.multiply(np.sign(agg_predicts) != train_label, np.ones([sample_num, 1]))
            error_rate = error_rate.sum()/sample_num

            if error_rate == 0:
                break
            self.classifier_set = weak_classifier
        return weak_classifier


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

        for classifier in self.classifier_set:
            alpha = classifier[0]
            clf = classifier[1]
            base_prediction = alpha * clf.predict(test_data)
            probability += base_prediction

        self.prediction = np.sign(probability)
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
        pickle.dump(self.classifier_set, f)
        f.close()

    '''
    Function:  load
    Description: load the model 
    Input:  filename    dataType: str   description: the path to save model
    Output: self        dataType: obj   description: the trained model
    '''
    def load(self, filename):
        f = open(filename)
        self.classifier_set = pickle.load(f)
        return self
