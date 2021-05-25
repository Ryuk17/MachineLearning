"""
@ Filename:       DimensionReduction.py
@ Author:         Ryuk
@ Create Date:    2019-06-02   
@ Update Date:    2019-06-06
@ Description:    Implement DimensionReduction
"""
import numpy as np
import pickle
import preProcess

class PCA:
    def __init__(self, norm_type="Standardization", rate=0.9):
        self.norm_type = norm_type
        self.matrix = None
        self.contribute_rate = None
        self.acc_contribute_rate = None
        self.rate = rate

    '''
       Function:  train
       Description: train the model
       Input:  train_data       dataType: ndarray   description: features
       Output: self             dataType: obj       description: the trained model
       '''
    def train(self, train_data):
        # decentration
        data = train_data - train_data.mean(axis=0)

        # calculate the eigenvalue and eigenvector of covariance matrix
        covariance_matrix = np.cov(data, rowvar=False)
        eigenvalue, eigenvector = np.linalg.eig(covariance_matrix)
        index = np.argsort(-eigenvalue)
        eigenvalue = eigenvalue[index]
        eigenvector = eigenvector[:, index]

        # calculate contribute rate
        contribute_rate = np.zeros(len(index))
        acc_contribute_rate = np.zeros(len(index))
        value_sum = eigenvalue.sum()
        sum = 0
        k = 0
        for i in range(len(eigenvalue)):
            sum = sum + eigenvalue[i]
            contribute_rate[i] = eigenvalue[i]/value_sum
            acc_contribute_rate[i] = sum/value_sum
            if (acc_contribute_rate[i-1] < self.rate) and (acc_contribute_rate[i] >= self.rate):
                k = i
        self.contribute_rate = contribute_rate
        self.acc_contribute_rate = acc_contribute_rate

        matrix = np.mat(eigenvector)[:, k]
        self.matrix = matrix
        return self

    '''
       Function:  transformData
       Description: transform data
       Input:  data                     dataType: ndarray   description: original data
       Output: transformed_data         dataType: ndarray   description: transformed data 
       '''
    def transformData(self, data):
        data = data - data.mean(axis=0)
        transformed_data = np.dot(data, self.matrix)
        return transformed_data

    '''
        Function:  save
        Description: save the model as pkl
        Input:  filename    dataType: str   description: the path to save model
        '''

    def save(self, filename):
        f = open(filename, 'w')
        pickle.dump(self.matrix, f)
        f.close()

    '''
    Function:  load
    Description: load the model 
    Input:  filename    dataType: str   description: the path to save model
    Output: self        dataType: obj   description: the trained model
    '''

    def load(self, filename):
        f = open(filename)
        self.matrix = pickle.load(f)
        return self


class LDA:
    def __init__(self, norm_type="Standardization", rate=0.9):
        self.norm_type = norm_type
        self.matrix = None
        self.contribute_rate = None
        self.acc_contribute_rate = None
        self.rate = rate

    '''
       Function:  train
       Description: train the model
       Input:  train_data       dataType: ndarray   description: features
       Output: self             dataType: obj       description: the trained model
       '''
    def train(self, data, label):
        # Normalization
        if self.norm_type == "Standardization":
            data = preProcess.Standardization(data)
        else:
            data = preProcess.Normalization(data)
        unique_label = np.unique(label)
        mu = np.mean(data, axis=0)
        # St = np.dot((data - mu).T, data - mu)

        Sw = 0
        Sb = 0
        for c in unique_label:
            index = np.where(label == c)
            Ni = len(index)
            xi = data[index]
            mui = np.mean(xi, axis=0)

            # calculate Sw
            Si = np.dot((xi - mui).T, xi - mui)
            Sw = Sw + Si

            # calculate Sb
            delta = np.expand_dims(mu - mui, axis=1)
            Sb = Sb + Ni * np.dot(delta, delta.T)

        # calculate the eigenvalue, eigenvector of Sw-1 * Sb
        temp = np.dot(np.linalg.inv(Sw), Sb)
        eigenvalue, eigenvector = np.linalg.eig(np.dot(np.linalg.inv(Sw), Sb))

        index = np.argsort(-eigenvalue)
        eigenvalue = eigenvalue[index]
        eigenvector = eigenvector[:, index]
        # calculate contribute rate
        contribute_rate = np.zeros(len(index))
        acc_contribute_rate = np.zeros(len(index))
        value_sum = eigenvalue.sum()
        sum = 0
        k = 0
        for i in range(len(eigenvalue)):
            sum = sum + eigenvalue[i]
            contribute_rate[i] = eigenvalue[i] / value_sum
            acc_contribute_rate[i] = sum / value_sum
            if (acc_contribute_rate[i - 1] < self.rate) and (acc_contribute_rate[i] >= self.rate):
                k = i

        self.contribute_rate = contribute_rate
        self.acc_contribute_rate = acc_contribute_rate

        matrix = np.mat(eigenvector)[:, k]
        self.matrix = matrix
        return self

    '''
       Function:  transformData
       Description: transform data
       Input:  data                     dataType: ndarray   description: original data
       Output: transformed_data         dataType: ndarray   description: transformed data 
       '''
    def transformData(self, data):
        transformed_data = np.dot(data, self.matrix)
        return transformed_data
