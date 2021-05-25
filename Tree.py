"""
@ Filename:       Tree.py
@ Author:         Ryuk
@ Create Date:    2019-05-16   
@ Update Date:    2019-05-16 
@ Description:    Classification and regression tree
"""

import numpy as np
import operator as op
import preProcess
import math
import pickle


class DecisionNode:
    def __init__(self, index=-1, value=None, results=None, right_tree=None, left_tree=None):
        self.index = index                    # the index of feature
        self.value = value                    # the value of the feature with index
        self.results = results                # current decision result
        self.right_tree = right_tree
        self.left_tree = left_tree


class DecisionTree:
    def __init__(self, norm_type="Normalization", t=1e-5):
        self.norm_type = norm_type
        self.t = t                          # the threshold of information gain
        self.prediction = None
        self.probability = None
        self.tree_node = None

    '''
    Function:  uniqueCount
    Description: calculate the count of unique labels
    Input:  labels             dataType: ndarray     description: labels of data
    Output: label_count        dataType: dictionary  description: [label, count]
    '''
    def uniqueCount(self, labels):
        label_count = {}
        for i in range(len(labels)):
            label_count[labels[i]] = label_count.get(labels[i], 0) + 1
        return label_count

    '''
    Function:  getEntropy
    Description: calcuate the Shannon entropy of the input data
    Input:  labels             dataType: ndarray   description: labels of data
    Output: entropy            dataType:       description: 
    '''
    def getEntropy(self, labels):
        labels_num = len(labels)
        label_count = self.uniqueCount(labels)

        entropy = 0.0
        for j in label_count:
            prop = label_count[j]/labels_num
            entropy = entropy + (-prop*math.log(prop, 2))

        return entropy

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
    Function:  createDecisionTree
    Description: create decision tree by ID3
    Input:  data            dataType: ndarray   description:  [feature,label]
    Output: self.tree_node  dataType: ndarray   description:  classification tree node
    '''
    def createDecisionTree(self, data):
        # if there is no feature in data, stop division
        if len(data) == 0:
            self.tree_node = DecisionNode()
            return self.tree_node

        best_gain = 0.0
        best_criteria = None
        best_set = None

        feature_num = len(data[0]) - 1
        sample_num = len(data[:, -1])
        init_entropy = self.getEntropy(data[:, -1])

        # get the best division
        for i in range(feature_num):
            uniques = np.unique(data[:, i])
            for value in uniques:
                left_set, right_set = self.divideData(data, i, value)
                # calcuate information gain
                ratio = float(len(left_set)/sample_num)
                if ratio == 0.0:
                    info_gain = init_entropy - (1 - ratio) * self.getEntropy(right_set[:, -1])
                elif ratio == 1.0:
                    info_gain = init_entropy - ratio*self.getEntropy(left_set[:, -1])
                else:
                    info_gain = init_entropy - ratio * self.getEntropy(left_set[:, -1]) - (1 - ratio) * self.getEntropy(right_set[:, -1])
                if info_gain > best_gain:
                    best_gain = info_gain
                    best_criteria = (i, value)
                    best_set = (left_set, right_set)

        # create the decision tree
        if best_gain < self.t:
            self.tree_node = DecisionNode(results=self.uniqueCount(data[:, -1]))
            return self.tree_node
        else:
            ltree = self.createDecisionTree(best_set[0])
            rtree = self.createDecisionTree(best_set[1])
            self.tree_node = DecisionNode(index=best_criteria[0], value=best_criteria[1], left_tree=ltree, right_tree=rtree)
            return self.tree_node

    '''
    Function:  vote
    Description: return the label of the majority  
    Input:  labels    dataType: ndarray   description:  labels
    Output: pred      dataType: int       description:  prediction label of input vector
    '''
    def vote(self, labels):
        label_count = {}
        # get the counts of each label
        for c in labels:
            label_count[c] = label_count.get(c, 0) + 1
        # get the labels of the majority
        predition = sorted(label_count.items(), key=op.itemgetter(1), reverse=True)
        pred = predition[0][0]
        return pred


    '''
    Function:  train
    Description: train the model
    Input:  train_data       dataType: ndarray   description: features
            train_label      dataType: ndarray   description: labels
    Output: self             dataType: obj       description: the trained model
    '''
    def train(self,train_data, train_label):
        if self.norm_type == "Standardization":
            train_data = preProcess.Standardization(train_data)
        else:
            train_data = preProcess.Normalization(train_data)

        train_label = np.expand_dims(train_label, axis=1)
        data = np.hstack([train_data, train_label])

        self.tree_node = self.createDecisionTree(data)
        #self.printTree(self.tree_node)
        return self

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
            result = sorted(result.items(), key=op.itemgetter(1), reverse=True)
            prediction[i] = result[0][0]
            #probability[i] = result[0][1]/(result[0][1] + result[1][1])
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
        if tree.results != None:
            return tree.results
        else:
            value = sample[tree.index]
            branch = None
            if value >= tree.value:
                branch = tree.right_tree
            else:
                branch = tree.left_tree
            return self.classify(sample, branch)

    '''
       Function:  printTree
       Description: show the structure of the decision tree
       Input:  tree        dataType: DecisionNode    description: decision tree
    '''
    def printTree(self, tree):
        # leaf node
        if tree.results != None:
            print(str(tree.results))
        else:
            # print condition
            print(str(tree.index) + ":" + str(tree.value) + "? ")
            # print subtree
            print("R->", self.printTree(tree.right_tree))
            print("L->", self.printTree(tree.left_tree))

    '''
    Function:  showDetectionResult
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


class RegressionNode():
    def __init__(self, index=-1, value=None, result=None, right_tree=None, left_tree=None):
        self.index = index
        self.value = value
        self.result = result
        self.right_tree = right_tree
        self.left_tree = left_tree


class RegressionTree:
    def __init__(self, norm_type="Normalization", error_threshold=1, N=4, alpha=0.01):
        self.norm_type = norm_type
        self.error_threshold = error_threshold  # the threshold of error
        self.N = N                              # the least number of sample for split
        self.alpha = alpha
        self.tree_node = None
        self.prediction = None

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
                right_set.append(temp)
            else:
                left_set.append(temp)
        return np.array(left_set), np.array(right_set)

    '''
       Function:  getVariance
       Description: get the variance of the regression value, in page of 68 Eq.(5.19)
       Input:  data            dataType: ndarray      description:  feature and value, the last column is value 
       Output: variance        dataType: ndarray      description:  variance 
       '''
    def getVariance(self, data):
        variance = np.var(data)
        return variance*len(data)

    '''
       Function:  getMean
       Description: get the mean of the regression value,in page of 68 Eq.(5.17)
       Input:  data            dataType: ndarray      description:  feature and value, the last column is value 
       Output: mean            dataType: ndarray      description:  mean
       '''
    def getMean(self, data):
        mean = np.mean(data)
        return mean

    '''
       Function:  createRegressionTree
       Description: create  regression tree
       Input:  data          dataType: ndarray      description:  training set
       Output: w             dataType: ndarray      description: weights
       '''
    def createRegressionTree(self, data):
        # if there is no feature
        if len(data) == 0:
            self.tree_node = RegressionNode(result=self.getMean(data[:, -1]))
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

        if best_set is None:
            self.tree_node = RegressionNode(result=self.getMean(data[:, -1]))
            return self.tree_node
        # if the descent of error is small enough, return the mean of the data
        elif abs(initial_error - best_error) < self.error_threshold:
            self.tree_node = RegressionNode(result=self.getMean(data[:, -1]))
            return self.tree_node
        # if the split data is small enough, return the mean of the data
        elif len(best_set[0]) < self.N or len(best_set[1]) < self.N:
            self.tree_node = RegressionNode(result=self.getMean(data[:, -1]))
            return self.tree_node
        else:
            ltree = self.createRegressionTree(best_set[0])
            rtree = self.createRegressionTree(best_set[1])
            self.tree_node = RegressionNode(index=best_criteria[0], value=best_criteria[1], left_tree=ltree, right_tree=rtree)
            return self.tree_node

    '''
       Function:  train
       Description: train the model
       Input:  train_data       dataType: ndarray   description: features
               train_label      dataType: ndarray   description: labels
       Output: self             dataType: obj       description: the trained model
       '''
    def train(self, train_data, train_label, pruning=False, val_data=None):
        # if self.norm_type == "Standardization":
        #     train_data = preProcess.Standardization(train_data)
        # else:
        #     train_data = preProcess.Normalization(train_data)

        train_label = np.expand_dims(train_label, axis=1)
        data = np.hstack([train_data, train_label])

        self.tree_node = self.createRegressionTree(data)
        #self.printTree(self.tree_node)
        if pruning:
            self.tree_node = self.pruning(self.tree_node, val_data)
        return self

    '''
       Function:  printTree
       Description: show the structure of the decision tree
       Input:  tree        dataType: DecisionNode    description: decision tree
    '''
    def printTree(self, tree):
        # leaf node
        if tree.result != None:
            print(str(tree.result))
        else:
            # print condition
            print(str(tree.index) + ":" + str(tree.value))
            # print subtree
            print("R->", self.printTree(tree.right_tree))
            print("L->", self.printTree(tree.left_tree))

    '''
     Function:  predict
     Description: predict the testing set 
     Input:  train_data       dataType: ndarray   description: features
             prob             dataType: bool      description: return probaility of label
     Output: prediction       dataType: ndarray   description: the prediction results for testing set
     '''
    def predict(self, test_data):
        # Normalization
        # if self.norm_type == "Standardization":
        #     test_data = preProcess.Standardization(test_data)
        # else:
        #     test_data = preProcess.Normalization(test_data)

        test_num = test_data.shape[0]
        prediction = np.zeros([test_num, 1])
        for i in range(test_num):
            prediction[i] = self.classify(test_data[i, :], self.tree_node)
        self.prediction = prediction

        return prediction

    '''
          Function:  classify
          Description: predict the testing set 
          Input:  sample      dataType: ndarray     description: input vector to be classified
          Output: label       dataType: ndarray     description: the prediction results of input
       '''
    def classify(self, sample, tree):
        if tree.result is not None:
            return tree.result
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
    def pruning(self, tree, val_data):
        if len(val_data) == 0:
            self.tree_node = RegressionNode(result=(tree.left_tree.result + tree.right_tree.result)/2)
            return self.tree_node

        left, right = self.divideData(val_data, tree.index, tree.value)
        if tree.left_tree is not None:
            self.pruning(tree.left_tree, left)
        if tree.right_tree is not None:
            self.pruning(tree.right_tree, right)

        # if there only exist two leaves node
        if tree.left_tree is None and tree.right_tree is None:
            left, right = self.divideData(val_data, tree.index, tree.value)
        return tree

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

