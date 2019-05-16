"""
@ Filename:       Cluster.py
@ Author:         Danc1elion
@ Create Date:    2019-05-15   
@ Update Date:    2019-05-16
@ Description:    Implement Cluster
"""
import sys
import numpy as np
import preProcess
import pickle
import random
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, norm_type="Normalization", k=5, distance_type="Euclidean"):
        self.norm_type = norm_type
        self.k = k
        self.distance_type = distance_type
        self.centers = None             # cluster cent
        self.distances = None             # distance between sample and cluster

    '''
      Function:  calcuateDistance
      Description: calcuate the distance between input vector and train data
      Input:  x1      dataType: ndarray   description: input vector
              x2      dataType: ndarray   description: input vector
      Output: d       dataType: float     description: distance between input vectors
      '''
    def calculateDistance(self, x1, x2):
        d = 0
        if self.distance_type == "Euclidean":
            d = np.sqrt(np.power(x1 - x2, 2))
        elif self.distance_type == "Cosine":
            d =  np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))
        elif self.distance_type == "Manhattan":
            d = np.sum(x1 - x2)
        else:
            print("Error Type!")
            sys.exit()
        return d

    '''
      Function:  createCenter
      Description: create cluster center
      Input:  train_data      dataType: ndarray   description: input vector
      Output: centers         dataType: ndarray   description: cluster centers
      '''
    def createCenter(self, train_data):
        feature_dim = np.shape(train_data)[1]
        centers = np.zeros([self.k, feature_dim])
        for i in range(feature_dim):
            min_value = np.min(train_data[:, i], axis=1)
            max_value = np.max(train_data[:, i], axis=1)
            centers[:, i] = min_value + (max_value - min_value) * np.random.rand(self.k, 1)  # generate a cluster center
        return centers

    '''
      Function:  kmeans
      Description: normal kmeans algorithm
      Input:  train_data           dataType: ndarray    description: features
      Output: self                 dataType: obj        description: trained model
          '''
    def kmeans(self, train_data, k):
        if self.norm_type == "Standardization":
            train_data = preProcess.Standardization(train_data)
        else:
            train_data = preProcess.Normalization(train_data)

        sample_num = len(train_data)
        distances = np.zeros([sample_num, 2])                      # (index, distance)
        centers = self.createCenter(train_data)
        flag = True                                                # If True, update cluster_center
        while flag:
            flag = False
            for i in range(sample_num):
                min_index = -1
                min_distance = np.inf

                # find the minimum distance between sample and cluster_center
                for j in range(k):
                    distance = self.calculateDistance(centers[j, :], train_data[i, :])
                    if distance < min_distance:
                        min_distance = distance
                        min_index = j

                # update center_distance
                if distances[i, 0] != min_index:
                    distances[i, 0] = min_index
                    distances[i, 1] = np.power(min_distance, 2)
                    flag = True

            # update cluster_center by calculate the mean of each cluster
            for m in range(k):
                current_cluster = train_data[distances[:, 0] == m]              # find the samples belong to the m-th cluster center
                centers[m, :] = np.mean(current_cluster, axis=0)
        return centers, distances

    '''
      Function:  biKmeans
      Description: binary kmeans algorithm
      Input:  train_data           dataType: ndarray    description: features
      Output: self                 dataType: obj        description: trained model
          '''
    def biKmeans(self, train_data):
        if self.norm_type == "Standardization":
            train_data = preProcess.Standardization(train_data)
        else:
            train_data = preProcess.Normalization(train_data)

        sample_num = len(train_data)
        distances = np.zeros([sample_num, 2])                                  # (index, distance)
        initial_center = np.mean(train_data, axis=0)                           # initial cluster #shape (1, feature_dim)
        centers = [initial_center]                                             # cluster list

        # clustering with the initial cluster center
        for i in range(sample_num):
            distances[i, 1] = np.power(self.calculateDistance(train_data[i, :], initial_center), 2)

        # generate cluster centers
        while len(centers) < self.k:

            min_SSE  = np.inf
            best_index = None                                                   # index of cluster for best split
            best_centers = None                                                 # best the cluster center
            best_distances = None                                                # the distance between samples and cluster center

            # find the best split
            for j in range(len(centers)):
                centerj_data = train_data[distances[:, 0] == j]                  # find the samples belong to the j-th center
                split_centers, split_distances = self.kmeans(centerj_data, 2)    # clustering the samples belong to j-th center into two cluster
                split_SSE = np.sum(split_distances[:, 1]) ** 2                   # calculate the distance for after clustering
                other_distances = distances[distances[:, 0] != j]                # the samples don't belong to j-th center
                other_SSE = np.sum(other_distances[:, 1]) ** 2                   # calculate the distance don't belong to j-th center

                # save the best split result
                if (split_SSE + other_SSE) < min_SSE:
                    best_index = j                                               # the best split index
                    best_centers = split_centers                                 # best cluster centers
                    best_distances = split_distances                             # the corresponding distance
                    min_SSE = split_SSE + other_SSE

            # save the spilt data
            best_distances[best_distances[:, 0] == 1, 0] = len(centers)         #
            best_distances[best_distances[:, 0] == 0, 0] = best_index           #

            centers[best_index] = best_centers[0, :]
            centers.append(best_centers)
            distances[np.nonzero(distances[:, 0] == best_index)[0], :] = best_distances

            return centers, distances



