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
    def __init__(self, norm_type="Normalization", k=5, distance_type="Euclidean", cluster_type="KMeans++"):
        self.norm_type = norm_type
        self.k = k
        self.distance_type = distance_type
        self.cluster_type = cluster_type
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
            d = np.sqrt(np.power(np.sum(x1 - x2, axis=1), 2))
        elif self.distance_type == "Cosine":
            d = np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))
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
            min_value = np.min(train_data[:, i])
            max_value = np.max(train_data[:, i])
            temp = min_value + (max_value - min_value) * np.random.rand(self.k)
            centers[:, i] = temp                # generate a cluster center
        return centers

    '''
      Function:  adjustCluster
      Description: adjust cluster when the cluster determined 
      Input:  centers         dataType: ndarray   description: cluster centers
              distances       dataType: ndarray   description: distance between sample and its corresponding cluster(cluster, distance)
              train_data      dataType: ndarray   description: train data
              k               dataType: int       description: the number of cluster 
      Output: centers         dataType: ndarray   description: cluster centers
              distances       dataType: ndarray   description: distance between sample and its corresponding cluster(cluster, distance)
      '''
    def adjustCluster(self, centers, distances, train_data, k):
        sample_num = len(train_data)
        flag = True  # If True, update cluster_center
        while flag:
            flag = False
            d = np.zeros([sample_num, len(centers)])
            for i in range(len(centers)):
                # calculate the distance between each sample and each cluster center
                d[:, i] = self.calculateDistance(train_data, centers[i])

            # find the minimum distance between each sample and each cluster center
            old_label = distances[:, 0].copy()
            distances[:, 0] = np.argmin(d, axis=1)
            distances[:, 1] = np.min(d, axis=1) ** 2
            if np.sum(old_label - distances[:, 0]) != 0:
                flag = True

            # for i in range(sample_num):
            #     min_index = -1
            #     min_distance = np.inf
            #
            #     # find the minimum distance between sample and cluster_center
            #     for j in range(k):
            #         print(j)
            #         distance = self.calculateDistance(train_data[i, :], centers[j, :])
            #         if distance < min_distance:
            #             min_distance = distance
            #             min_index = j
            #
            #     # update center_distance
            #     if distances[i, 0] != min_index:
            #         distances[i, 0] = min_index
            #         distances[i, 1] = np.power(min_distance, 2)
            #         flag = True

            # update cluster_center by calculate the mean of each cluster
            for m in range(k):
                current_cluster = train_data[distances[:, 0] == m]  # find the samples belong to the m-th cluster center
                centers[m, :] = np.mean(current_cluster, axis=0)
            return centers, distances

    '''
      Function:  kmeans
      Description: normal kmeans algorithm
      Input:  train_data           dataType: ndarray    description: features
      Output: centers         dataType: ndarray   description: cluster centers
              distances       dataType: ndarray   description: distance between sample and its corresponding cluster(cluster, distance)
          '''
    def kmeans(self, train_data, k):
        sample_num = len(train_data)
        distances = np.zeros([sample_num, 2])                      # (index, distance)
        centers = self.createCenter(train_data)
        centers, distances = self.adjustCluster(centers, distances, train_data, self.k)
        return centers, distances

    '''
      Function:  biKmeans
      Description: binary kmeans algorithm
      Input:  train_data      dataType: ndarray    description: features
      Output: centers         dataType: ndarray   description: cluster centers
              distances       dataType: ndarray   description: distance between sample and its corresponding cluster(cluster, distance)
          '''
    def biKmeans(self, train_data):
        sample_num = len(train_data)
        distances = np.zeros([sample_num, 2])                                  # (index, distance)
        initial_center = np.mean(train_data, axis=0)                           # initial cluster #shape (1, feature_dim)
        centers = [initial_center]                                             # cluster list

        # clustering with the initial cluster center

        distances[:, 1] = np.power(self.calculateDistance(train_data, initial_center), 2)

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
            best_distances[best_distances[:, 0] == 1, 0] = len(centers)         # samples of cluster 1 denote as a new cluster
            best_distances[best_distances[:, 0] == 0, 0] = best_index           # samples of cluster 0 denote as the split-index cluster

            centers[best_index] = best_centers[0, :]                            # update cluster
            centers.append(best_centers[1, :])                                  # add a new cluster
            distances[distances[:, 0] == best_index, :] = best_distances        # save the distances

        return centers, distances

    '''
      Function:  kmeansplusplus
      Description:  kmeans++ algorithm
      Input:  train_data      dataType: ndarray    description: features
      Output: centers         dataType: ndarray   description: cluster centers
              distances       dataType: ndarray   description: distance between sample and its corresponding cluster(cluster, distance)
          '''
    def kmeansplusplus(self,train_data):
        sample_num = len(train_data)
        distances = np.zeros([sample_num, 2])                                  # (index, distance)

        # randomly select a sample as the initial cluster
        initial_center = train_data[np.random.randint(0, sample_num-1)]
        centers = [initial_center]

        while len(centers) < self.k:
            d = np.zeros([sample_num, len(centers)])
            for i in range(len(centers)):
                # calculate the distance between each sample and each cluster center
                d[:, i] = self.calculateDistance(train_data, centers[i])

            # find the minimum distance between each sample and each cluster center
            distances[:, 0] = np.argmin(d, axis=1)
            distances[:, 1] = np.min(d, axis=1)

            # Roulette Wheel Selection
            prob = np.power(distances[:, 1], 2)/np.sum(np.power(distances[:, 1], 2))
            index = np.argmax(prob)
            new_center = train_data[index, :]
            centers.append(new_center)

        # adjust cluster
        centers = np.array(centers)   # transform form list to array
        centers, distances = self.adjustCluster(centers, distances, train_data, self.k)
        return centers, distances

    '''
    Function:  cluster
    Description: train the model
    Input:  train_data      dataType: ndarray   description: features
    Output: centers         dataType: ndarray   description: cluster centers
            distances       dataType: ndarray   description: distance between sample and its corresponding cluster(cluster, distance)
    '''
    def cluster(self, train_data, display="True"):
        if self.norm_type == "Standardization":
            train_data = preProcess.Standardization(train_data)
        else:
            train_data = preProcess.Normalization(train_data)

        if self.cluster_type == "KMeans":
            self.centers, self.distances = self.kmeans(train_data, self.k)
        elif self.cluster_type == "biKMeans":
            self.centers, self.distances = self.biKmeans(train_data)
        elif self.cluster_type == "KMeans++":
            self.centers, self.distances = self.kmeansplusplus(train_data)
        else:
            print("Wrong cluster type!")
            sys.exit()
        if display:
            self.plotResult(train_data)
        return self.distances[:, 0]

    '''
    Function:  plotResult
    Description: show the clustering result
    '''
    def plotResult(self, train_data):
        plt.scatter(train_data[:, 0], train_data[:, 1], c=self.distances[:, 0])
        plt.show()

    '''
      Function:  save
      Description: save the model as pkl
      Input:  filename    dataType: str   description: the path to save model
      '''

    def save(self, filename):
        f = open(filename, 'w')
        model = {'centers': self.centers, 'distances': self.distances}
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
        self.centers = model['centers']
        self.distances = model['distances']
        return self
