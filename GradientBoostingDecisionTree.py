"""
@ Filename:       GradientBoostingDecisionTree.py
@ Author:         Ryuk
@ Create Date:    2019-07-09   
@ Update Date:    2019-07-10
@ Description:    Implement GradientBoostingDecisionTree
"""

import numpy as np
from TreeRegression import RegressionTree
import pickle

class GBDTClassifier:
    def __init__(self, tree_num=10):
        self.tree_num = tree_num




class GBDTRegression:
    def __init__(self, tree_num=10, error_threshold=1, N=4, alpha=0.01, iterations=100):
        self.tree_num = tree_num
        self.error_threshold = error_threshold
        self.N = N
        self.alpha = alpha
        self.trees = []
        self.gamma = []                    # multiplier for each model
        self.residual = None
        self.iterations = iterations       # iterations for gamma
        self.last_prediction = None
        self.prediction = None

    '''
        Function:  initializeModel
        Description: initialize the model
        Input:  train_label      dataType: ndarray   description: train_label
    '''
    def initializeModel(self, train_label):
        x = np.mean(train_label)
        for i in range(self.iterations):
            error = train_label - x
            x = x - self.alpha * error
        self.residual = train_label - x
        self.last_prediction = x
        self.trees.append(x)
        self.gamma.append(1)

    '''
        Function: getGamma
        Description: get gamma
        Input:  train_data       dataType: ndarray   description: features
                prediction       dataType: ndarray   description: prediction
    '''
    def getGamma(self, train_label, last_prediction, current_prediction):
        gamma = np.mean(train_label)
        for i in range(self.iterations):
            error = train_label - last_prediction - gamma * current_prediction
            gamma = gamma - self.alpha*error
        self.residual = train_label - last_prediction - gamma * current_prediction
        self.last_prediction = last_prediction + gamma * current_prediction
        self.gamma.append(gamma)

    '''
        Function:  train
        Description: train the model
        Input:  train_data       dataType: ndarray   description: features
                train_label      dataType: ndarray   description: labels
        Output: self             dataType: obj       description: the trained model
    '''
    def train(self, train_data, train_label):
        # initialize
        self.initializeModel(train_label)

        # train
        for i in range(self.tree_num):
            clf = RegressionTree(self.error_threshold, self.N, self.alpha)
            clf.train(train_data, self.residual)
            prediction = clf.predict(train_data)
            self.trees.append(clf)
            self.getGamma(train_label, self.last_prediction, prediction)
        return self

    '''
     Function:  predict
     Description: predict the testing set 
     Input:  test_data        dataType: ndarray   description: features
     Output: prediction       dataType: ndarray   description: the prediction results for testing set
     '''
    def perdict(self, test_data):
        prediction = np.zeros(len(test_data))
        for i in range(self.tree_num):
            if i == 0:
                prediction += self.gamma * self.trees[i]
            else:
                clf_prediction = self.trees[i].predict(test_data)
                prediction += self.gamma * clf_prediction

        self.prediction = prediction
        return prediction

    '''
      Function:  save
      Description: save the model as pkl
      Input:  filename    dataType: str   description: the path to save model
      '''

    def save(self, filename):
        f = open(filename, 'w')
        model = {'trees':self.trees, 'gamma': self.gamma}
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
        self.trees = model['trees']
        self.gamma = model['gamma']
        return self
