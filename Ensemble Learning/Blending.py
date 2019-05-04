"""
@ Filename:       Blending .py
@ Author:         Danc1elion
@ Create Date:    2019-05-04
@ Update Date:    2019-05-04
@ Description:    Implement Blending
"""

from sklearn.model_selection import StratifiedKFold, train_test_split
from Perceptron import *
import numpy as np
import preProcess
import pickle
import random


class BlendingClassifier:
    def __init__(self, norm_type="Normalization", classifier_set=None):
        self.norm_type = norm_type
        self.classifier_set = classifier_set
        self.k = len(self.classifier_set)       # the number of classifiers
        self.layer1_classifier_set = None
        self.layer2_classifier = None
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

        train_data1, train_data2, train_label1, train_label2 = train_test_split(train_data, train_label, test_size=0.5, random_state=2019)
        # train set in the second layer
        train_predict_feature = np.zeros((train_data2.shape[0], self.k))
        trained_model = []

        # the first layer in Blending
        for j, clf in enumerate(self.classifier_set):
            # train each submodel
            print(j, clf)
            clf.train(train_data1, train_label1)
            train_predict_feature[:, j] = clf.predict(train_data2)[:, 0]
            # save the trained model in the first layer
            trained_model.append(clf)

        # the second layer in Blending
        layer2_clf = PerceptronClassifier()
        layer2_clf.train(train_predict_feature, train_label2)

        self.layer1_classifier_set = trained_model
        self.layer2_classifier = layer2_clf

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

        test_predict_feature = np.zeros((test_data.shape[0], self.k))
        # the first layer in Blending
        for j, clf in enumerate(self.layer1_classifier_set):
            test_predict_feature[:, j] = clf.predict(test_data)[:, 0]

        # the second layer in Blending
        probability = self.layer2_classifier.predict(test_predict_feature)
        prediction = (probability > 0.5)*1

        self.probability = probability
        self.prediction = prediction
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
        accuarcy = sum(prediction == test_label) / len(test_label)
        return accuarcy

    '''
       Function:  save
       Description: save the model as pkl
       Input:  filename    dataType: str   description: the path to save model
       '''

    def save(self, filename):
        f = open(filename, 'w')
        model = {'layer1_classifiers':self.layer1_classifier_set, 'layer2_classifier':self.layer2_classifier}
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
        self.layer1_classifier_set = model['layer1_classifiers']
        self.layer2_classifier = model['layer2_classifier']
        return self





