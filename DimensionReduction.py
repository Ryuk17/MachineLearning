"""
@ Filename:       DimensionReduction.py
@ Author:         Danc1elion
@ Create Date:    2019-06-02   
@ Update Date:    2019-06-03
@ Description:    Implement DimensionReduction
"""
import numpy as np

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



