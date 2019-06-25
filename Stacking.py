"""
@ Filename:       Stacking.py
@ Author:         Danc1elion
@ Create Date:    2019-05-05
@ Update Date:    2019-05-05
@ Description:    Implement Stacking
"""
from sklearn.model_selection import StratifiedKFold, train_test_split
from Perceptron import *
import numpy as np
import preProcess
import pickle
import random

class StackingClassifier:
    def __init__(self, norm_type="Normalization", classifier_set=None, fusion_type="Weighing",n_folds=5):
        self.norm_type = norm_type
        self.classifier_set = classifier_set
        self.k = len(self.classifier_set)       # the number of classifiers
        self.trained_classifier_set = None
        self.n_folds = n_folds                  # the number of fold for cross validation
        self.fusion_type = fusion_type          # fusion method in the second layer
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

        skf = StratifiedKFold(self.n_folds)
        prediction_feature = np.zeros((train_data.shape[0], len(self.classifier_set)))
        trained_model = []

        # the first layer in Stacking
        for j, clf in enumerate(self.classifier_set):
            # train each submodel
            subtrained_model = []
            # cross validation
            for (train_index, test_index) in skf.split(train_data, train_label):
                X_train, X_test = train_data[train_index], train_data[test_index]
                y_train, y_test = train_label[train_index], train_label[test_index]
                # train and save the model trained with S-si
                clf.train(X_train, y_train)
                subtrained_model.append(clf)
                # get the prediction feature for each sub model
                prediction_feature[test_index, j] = clf.predict(X_test)[:, 0]
            # save the models
            trained_model.append(subtrained_model)

        self.trained_classifier_set = trained_model
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

        pre_prediction = np.zeros((test_data.shape[0], self.n_folds))
        # the first layer in Stacking
        for j, sub_model in enumerate(self.trained_classifier_set):
            sub_prediction_feature = np.zeros((test_data.shape[0], self.n_folds))
            i = 0
            for clf in sub_model:
                sub_prediction_feature[:, i] = clf.predict(test_data)[:, 0]
                i = i + 1
            pre_prediction[:, j] = sub_prediction_feature.mean(1)

        test_num = test_data.shape[0]
        prediction = np.zeros([test_num, 1])
        probability = np.zeros([test_num, 1])
        # the second layer in Stacking
        if self.fusion_type == "Averaging":
            probability = pre_prediction.mean(1)
        elif self.fusion_type == "Voting":
            probability = np.sum(pre_prediction, axis=1)/self.k
        elif self.fusion_type == "Weighing":
            w = [i/i.sum() for i in pre_prediction]
            probability = np.sum(np.multiply(pre_prediction, w), axis=1)

        prediction = (probability > 0.5) * 1
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
        # test_label = np.expand_dims(test_label, axis=1)
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
        pickle.dump(self.trained_classifier_set, f)
        f.close()

    '''
    Function:  load
    Description: load the model 
    Input:  filename    dataType: str   description: the path to save model
    Output: self        dataType: obj   description: the trained model
    '''

    def load(self, filename):
        f = open(filename)
        self.trained_classifier_set = pickle.load(f)
        return self

