"""
@FileName: GMM.py
@Description: Implement GMM
@Author: Ryuk
@CreateDate: 2021/05/30
@LastEditTime: 2021/05/30
@LastEditors: Please set LastEditors
@Version: v0.1
"""

import numpy as np
import pickle
import preProcess
from tqdm import tqdm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class GaussianMixtureModel:
    def __init__(self, K, D=2, iterations=100, norm_type="Normalization"):
        self.norm_type = norm_type
        self.iterations = iterations
        self.K = K
        self.D = D
        self.N = 0
        self.alpha = np.random.dirichlet(np.ones(self.K))
        self.mu = np.random.rand(K, D)
        self.sigma = np.array([np.eye(self.D)] * K)
        self.gamma = None
        self.label = None

    '''
    Function:  GaussianPDF
    Description: generate gaussian distribution with given mu, sigma and x
    Input:  mu          dataType: ndarray   description: features
    Input:  sigma       dataType: ndarray   description: features
    Input:  x           dataType: ndarray   description: features
    Output: self        dataType: obj       description: the trained model
    '''
    def GaussianPDF(self, mu, sigma, x):
        gaussian = multivariate_normal(mu, sigma)
        return gaussian.pdf(x)

    '''
    Function:  train
    Description: train the model
    Input:  train_data       dataType: ndarray   description: features
    Output: self             dataType: obj       description: the trained model
    '''
    def train(self, train_data, plotResult=True):
        self.N = len(train_data)
        self.gamma = np.zeros([self.N, self.K])

        # if self.norm_type == "Standardization":
        #     train_data = preProcess.Standardization(train_data)
        # else:
        #     train_data = preProcess.Normalization(train_data)

        for i in tqdm(range(self.iterations)):
            # E-step
            for k in range(self.K):
                self.gamma[:,k] = self.GaussianPDF(self.mu[k], self.sigma[k], train_data)

            for j in range(self.N):
                self.gamma[j,:] = self.gamma[j,:] / np.sum(self.gamma[j,:])

            # M-step
            for k in range(self.K):
                gamma_sum = np.sum(self.gamma[:,k])
                self.mu[k] = np.sum(np.dot(self.gamma[None,:, k], train_data), axis=0) / gamma_sum
                self.sigma[k] = (train_data - self.mu[k]).T * np.multiply(np.mat(train_data - self.mu[k]), np.mat(self.gamma[:, k]).T) / gamma_sum
                self.alpha[k] = gamma_sum / self.N
        self.label = np.argmax(self.gamma, axis=1)

        if plotResult:
            self.plotResult(train_data)
        return self.label


    '''
    Function:  predict
    Description: predict the test data
    Input:  test_data        dataType: ndarray   description: features
    Output: label            dataType: ndarray   description: the predicted label
    '''
    def predict(self, test_data):
        self.N = len(test_data)
        self.gamma = np.zeros([self.N, self.K])

        for k in range(self.K):
            gamma_sum = np.sum(self.gamma[:,k])
            self.mu[k] = np.sum(np.dot(self.gamma[None,:, k], test_data), axis=0) / gamma_sum
            self.sigma[k] = (test_data - self.mu[k]).T * np.multiply(np.mat(test_data - self.mu[k]), np.mat(self.gamma[:, k]).T) / gamma_sum
            self.alpha[k] = gamma_sum / self.N
        self.label = np.argmax(self.gamma, axis=1)
        return self.label

    '''
    Function:  plotResult
    Description: show the clustering result
    '''
    def plotResult(self, train_data):
        plt.scatter(train_data[:, 0], train_data[:, 1], c=self.label)
        plt.title('GMM')
        plt.show()

    '''
         Function:  save
         Description: save the model as pkl
         Input:  filename    dataType: str   description: the path to save model
         '''
    def save(self, filename):
        f = open(filename, 'w')
        model = {'alpha': self.alpha, 'mu': self.mu, 'sigma': self.sigma}
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
        self.alpha = model['alpha']
        self.mu = model['mu']
        self.sigma = model['sigma']
        return self