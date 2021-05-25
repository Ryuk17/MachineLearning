"""
@ Filename:       RandomForest.py
@ Author:         Ryuk
@ Create Date:    2019-07-09   
@ Update Date:    2019-07-09 
@ Description:    Implement RandomForest
"""
import numpy as np
import operator as op
import pickle
from DecisionTree import DecisionTreeClassifier
from TreeRegression import RegressionTree

class RandomForestClassifier:
    def __init__(self, tree_num=10, alpha=1e-5):
        self.tree_num = tree_num
        self.alpha=alpha
        self.trees = []
        self.prediction = None
        self.probability = None

    '''
        Function:  boostrap
        Description: boostrap sampling and train a model
        Input:  train_data       dataType: ndarray   description: features
                train_label      dataType: ndarray   description: labels
                self             dataType: obj       description: the trained model
    '''
    def boostrap(self, train_data, train_label):
        index = np.random.randint(0, len(train_data), (len(train_data)))
        x = train_data[index]
        y = train_label[index]
        clf = DecisionTreeClassifier(t=self.alpha)
        clf.train(x, y)
        return clf

    '''
        Function:  train
        Description: train the model
        Input:  train_data       dataType: ndarray   description: features
                train_label      dataType: ndarray   description: labels
        Output: self             dataType: obj       description: the trained model
          '''
    def train(self, train_data, train_label):
        for i in range(self.tree_num):
            clf = self.boostrap(train_data, train_label)
            self.trees.append(clf)
        return self

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
     Function:  predict
     Description: predict the testing set 
     Input:  test_data       dataType: ndarray   description: features
     Output: prediction       dataType: ndarray   description: the prediction results for testing set
     '''
    def predict(self, test_data):
        labels = np.zeros([len(test_data), self.tree_num])
        for i in range(self.tree_num):
            clf = self.trees[i]
            labels[:, i] = clf.predict(test_data).reshape(len(test_data))

        prediction = np.zeros([len(test_data)])
        for j in range(len(labels)):
            prediction[j] = self.vote(labels[j,:])

        self.prediction = prediction
        return prediction

    '''
    Function:  showDetectionResult
    Description: show detection result
    Input:  test_data  dataType: ndarray   description: data for test
            test_label dataType: ndarray   description: labels of test data
    Output: accuracy   dataType: float     description: detection accuarcy
    '''
    def accuarcy(self, test_label):
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
        model = self.trees
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
        self.trees = pickle.load(f)
        return self


class RandomForestRegression:
    def __init__(self, tree_num=10, error_threshold=1,  N=4, alpha=0.01):
        self.sample_num = 0
        self.tree_num = tree_num
        self.trees = []
        self.error_threshold = error_threshold  # the threshold of error
        self.N = N                              # the least number of sample for split
        self.alpha = alpha
        self.tree_node = None
        self.prediction = None

    '''
        Function:  boostrap
        Description: boostrap sampling and train a model
        Input:  train_data       dataType: ndarray   description: features
                train_label      dataType: ndarray   description: labels
                self             dataType: obj       description: the trained model
    '''
    def boostrap(self, train_data, train_label):
        index = np.random.randint(0, self.sample_num, (self.sample_num))
        x = train_data[index]
        y = train_label[index]
        clf = RegressionTree(error_threshold=1,  N=4, alpha=0.01)
        clf.train(x, y)
        return clf

    '''
        Function:  train
        Description: train the model
        Input:  train_data       dataType: ndarray   description: features
                train_label      dataType: ndarray   description: labels
        Output: self             dataType: obj       description: the trained model
          '''
    def train(self, train_data, train_label):
        for i in range(self.tree_num):
            clf = self.boostrap(train_data, train_label)
            self.trees.append(clf)
        return self

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
     Function:  predict
     Description: predict the testing set 
     Input:  test_data        dataType: ndarray   description: features
     Output: prediction       dataType: ndarray   description: the prediction results for testing set
     '''
    def predict(self, test_data):
        labels = np.zeros([len(test_data), self.tree_num])
        for i in range(self.tree_num):
            labels[:,i] = self.trees[i].predict(test_data)

        prediction = np.mean(labels, axis=0)

        self.prediction = prediction
        return prediction

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

    '''
         Function:  save
         Description: save the model as pkl
         Input:  filename    dataType: str   description: the path to save model
         '''
    def save(self, filename):
        f = open(filename, 'w')
        model = self.trees
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
        self.trees = pickle.load(f)
        return self
