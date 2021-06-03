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

class GaussianMixtureModel:
    def __init__(self, K, D, iterations=10, norm_type="Normalization"):
        self.norm_type = norm_type
        self.iterations = iterations
        self.K = K
        self.D = D
        self.N = 0
        self.alpha = np.random.dirichlet(np.ones(self.K), size=self.K)
        self.mu = np.zeros((self.K, self.D))
        self.sigma = np.array([np.eye(self.D)] * K)
        self.gamma = None

    '''
    Function:  GaussianPDF
    Description: generate gaussian distribution with given mu, sigma and x
    Input:  mu      dataType: ndarray   description: features
    Input:  sigma       dataType: ndarray   description: features
    Input:  x       dataType: ndarray   description: features
    Output: self             dataType: obj       description: the trained model
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
    def train(self, train_data):
        self.N = len(train_data)
        self.gamma = np.zeros(self.N, self.K)

        if self.norm_type == "Standardization":
            train_data = preProcess.Standardization(train_data)
        else:
            train_data = preProcess.Normalization(train_data)

        for i in tqdm(range(self.iterations)):
            # E-step
            for k in range(self.K):
                self.gamma[:,k] = self.alpha[k] * self.GaussianPDF(self.mu[k], self.sigma[k], train_data)

            for j in range(self.N):
                self.gamma[j,:] = self.gamma[j,:] / np.sum(self.gamma[j,:])

            # M-step
            for k in range(self.K):
                gamma_sum = np.sum(self.gamma[:,k])

                self.mu[k] = np.sum(np.multiply(train_data, self.gamma[:, k]), axis=0) / gamma_sum
                self.sigma[k] = (train_data - self.mu[k]).T * np.multiply((train_data - self.mu[k]), self.gamma[:, k]) / gamma_sum
                self.alpha[k] = gamma_sum / self.N

    def predict(self, test_data):
        pass


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