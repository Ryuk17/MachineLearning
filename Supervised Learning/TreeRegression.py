"""
@ Filename:       TreeRegression.py
@ Author:         Danc1elion
@ Create Date:    2019-05-11
@ Update Date:    2019-05-11
@ Description:    Implement TreeRegression
"""

import numpy as np
import operator as op
import preProcess
import math
import pickle

class treeNode():
    def __init__(self, index=-1, value=None, result=None, right_tree=None, left_tree=None):
        self.index = index
        self.value = value
        self.result = result
        self.right_tree = right_tree
        self.left_tree = left_tree


class Regression:
    def __init__(self, norm_type="Normalization",iterations=100, error_threshold=0.01, N=4):
        self.norm_type = norm_type
        self.iterations = iterations
        self.error_threshold = error_threshold  # the threshold of error
        self.N = N                              # the least number of sample for split
        self.tree_node = None
        self.prediction = None
        self.probability = None

    '''
    Function:  divideData
    Description: divide data into two parts
    Input:  data               dataType: ndarray   description:  feature and labels
            index              dataType: int       description:  the column of feature
            value              dataType: float     description:  the value of feature
    Output: left_set           dataType: ndarray   description:  feature <= value
            right_set          dataType: ndarray   description:  feature > value
    '''
    def divideData(self, data, index, value):
        left_set = []
        right_set = []
        # select feature in index with value
        for temp in data:
            if temp[index] >= value:
                # delete this feature
                new_feature = np.delete(temp, index)
                right_set.append(new_feature)
            else:
                new_feature = np.delete(temp, index)
                left_set.append(new_feature)
        return np.array(left_set), np.array(right_set)

    '''
       Function:  getVariance
       Description: get the variance of the regression value, in page of 68 Eq.(5.19)
       Input:  data            dataType: ndarray      description:  feature and value, the last column is value 
       Output: variance        dataType: ndarray      description:  variance 
       '''
    def getVariance(self, data):
        variance = np.var(data[:, -1])
        return variance*len(data)

    '''
       Function:  getMean
       Description: get the mean of the regression value,in page of 68 Eq.(5.17)
       Input:  data            dataType: ndarray      description:  feature and value, the last column is value 
       Output: mean            dataType: ndarray      description:  mean
       '''
    def getMean(self, data):
        mean = np.var(data[:, -1])
        return mean

    '''
       Function:  createRegressionTree
       Description: create  regression tree
       Input:  data          dataType: ndarray      description:  training set
       Output: w             dataType: ndarray      description: weights
       '''
    def createRegressionTree(self, data):
        if len(data) == 0:
            self.tree_node = treeNode()
            return self.tree_node

        sample_num, feature_dim = np.shape(data)

        best_criteria = None
        best_error = np.inf
        best_set = None
        initial_error = self.getVariance(data)

        # get the best split feature and value
        for index in range(feature_dim - 1):
            uniques = np.unique(data[:, index])
            for value in uniques:
                left_set, right_set = self.divideData(data, index, value)
                if len(left_set) < self.N or len(right_set) < self.N:
                    continue
                new_error = self.getVariance(left_set) + self.getVariance(right_set)
                if new_error < best_error:
                    best_criteria = (index, value)
                    best_error = new_error
                    best_set = (left_set, right_set)

        # if the descent of error is small enough, return the mean of the data
        if abs(initial_error - best_error) < self.error_threshold:
            self.tree_node = treeNode(result=self.getMean(data[:, -1]))
            return self.tree_node
        # if the split data is small enough, return the mean of the data
        elif len(best_set[0]) < self.N or len(best_set[1]) < self.N:
            self.tree_node = treeNode(result=self.getMean(data[:, -1]))
            return self.tree_node
        else:
            ltree = self.createRegressionTree(best_set[0])
            rtree = self.createRegressionTree(best_set[1])
            self.tree_node = treeNode(index=best_criteria[0], value=best_criteria[1], left_tree=ltree, right_tree=rtree)
            return self.tree_node

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

        train_label = np.expand_dims(train_label, axis=1)
        data = np.hstack([train_data, train_label])

        self.tree_node = self.createRegressionTree(data)
        # self.printTree(self.tree_node)
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
            result = self.classify(test_data[i, :], self.tree_node)
            # probability[i] = result[0][1]/(result[0][1] + result[1][1])
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
    def classify(self, sample, tree):
        if tree.results is not None:
            return tree.results
        else:
            value = sample[tree.index]
            if value >= tree.value:
                branch = tree.right_tree
            else:
                branch = tree.left_tree
            return self.classify(sample, branch)

    '''
       Function:  pruning
       Description: pruning the regression tree
       Input:  test_data       dataType: ndarray   description: features
               test_label      dataType: ndarray   description: labels
       Output: self            dataType: obj       description: the trained model
       '''
    def pruning(self, tree, data, alpha):

        return 0







    '''
      Function:  save
      Description: save the model as pkl
      Input:  filename    dataType: str   description: the path to save model
      '''

    def save(self, filename):
        f = open(filename, 'w')
        pickle.dump(self.tree_node, f)
        f.close()

    '''
    Function:  load
    Description: load the model 
    Input:  filename    dataType: str   description: the path to save model
    Output: self        dataType: obj   description: the trained model
    '''

    def load(self, filename):
        f = open(filename)
        self.tree_node = pickle.load(f)
        return self


