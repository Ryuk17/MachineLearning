import numpy as np
import preProcess
import pickle
import random


class SVMClassifier:
    def __init__(self, norm_type="Normalization", C=200, kernel="rbf", threshold=10e-3, g=0.1, c=0, n=3, max_iteration=100):
        self.norm_type = norm_type
        self.prediction = None
        self.probability = None
        self.train_data = None
        self.train_label = None
        self.sample_num = None
        self.max_iteration = max_iteration             # max iteration of SMO
        self.K = None
        self.alphas = None
        self.w = None                                 # the weight of hyperplane
        self.b = None                                 # the bias of hyperplane
        self.errors = None                            # errors
        self.C = C                                    # penalty coefficient
        self.threshold = threshold                    # threshold of tolerant error
        self.kernel = kernel                          # kernel function
        self.g = g                                    # sigma for rbf, sigmoid poly
        self.n = n                                    # order of poly
        self.c = c                                    # bias of sigmoid poly


    '''
       Function:  labelTransformation
       Description: transform {0, 1} into {-1, 1}, list to ndarray
       Input: labels      dataType: List        description: original label
       Output: new_label  dataType: ndarray     description: new label
       '''
    def labelTransformation(self, labels):
        new_labels = np.zeros([len(labels), 1])
        for i in range(len(labels)):
            if labels[i] == 0:
                new_labels[i] = labels[i]
            else:
                new_labels[i] = labels[i]
        return new_labels

    '''
       Function:  calculateErrors 
       Description: calculate the prediction errors of the k-th sample  LiHang statistical learning P127 Eq. (7.105)
                    g(x) = sigma[ai*yi*K(xi,x))] + b
       Input: k           dataType: int         description: index of the k-th sample
       Output: Ek         dataType: float       description: prediction error of the k-th sample
       '''
    def calculateErrors(self, k):
        gap = np.dot(np.multiply(self.alphas, self.train_label).T, self.K[:, k]) + self.b
        Ek = gap - self.train_label[k]
        return Ek

    '''
       Function:  selectAlpha2Rand
       Description: select alpha2 
       Input: i           dataType: int         description: the index of the alpha1
       Output: j          dataType: int         description: the index of the alpha2
       '''
    def selectAlpha2Rand(self, i):
        j = i
        while j == i:
            j = random.randint(0, self.sample_num)
        return j

    '''
       Function:  selectAplha2 
       Description: select the second alpha by elicitation method in inner loop 
       Input: i           dataType: int         description: the index of the first alpha 
              Ei          dataType: float       description: the error of the first alpha
       Output: j          dataType: int         description: the index of the second alpha
               Ej         dataType: float       description: the error of the second alpha 
            
       '''
    def selectAplha2(self, i, Ei):
        max_k = -1
        max_delta = 0.0
        Ej = 0.0

        self.errors[i] = [1, Ei]
        valid_errors_index = np.nonzero(self.errors[:, 0])[0]     # get the nonzero value of the alpha
        if len(valid_errors_index) > 1:
            for k in valid_errors_index:
                if k == i:
                    continue
                Ek = self.calculateErrors(k)
                # print(self.calculateErrors(k))
                delta_e = abs(Ei - Ek)
                if delta_e > max_delta:                 # select j with the max Ei-Ej
                    max_k = k
                    max_delta = delta_e
                    Ej = Ek
            return max_k, Ej
        else:
            j = self.selectAlpha2Rand(i)
            Ej = self.calculateErrors(j)
        return j, Ej

    '''
       Function:  upadateError
       Description: update and save the perdiction errors
       Input: k           dataType: int         description: the index of the first alpha 
       '''
    def upadateError(self, k):
        Ek = self.calculateErrors(k)
        self.errors[k] = [1, Ek]  # 1 means valid

    '''
       Function:  upadateError
       Description: update and save the perdiction errors, in page of 127 Eq.(7.108)
       Input: alpha2      dataType: float       description: old alpha2 
              L           dataType: float       description: low border of alpha2 
              H           dataType: float       description: high border of alpha2 
        Output: alpha2    dataType: float       description: new alpha2 
       '''
    def updateAlpha2(self, alpha2, L, H):
        if alpha2 > H:
            alpha2 = H
        if L > alpha2:
            alpha2 = L
        return alpha2

    '''
       Function:  innerLoop
       Description: inner loop in Platt SMO
       Input: i           dataType: int         description: the index of the first alpha 
       '''
    def innerLoop(self, i):
        Ei = self.calculateErrors(i)
        # check KKT conditions
        if ((self.train_label[i] * Ei < -self.threshold) and (self.alphas[i] < self.C)) or ((self.train_label[i] * Ei > self.threshold) and (self.alphas[i] > 0)):

            j, Ej = self.selectAplha2(i, Ei)          # select alpha2 according to alpha1

            # copy alpha1 and alpha2
            old_alpha1 = self.alphas[i]
            old_alpha2 = self.alphas[j]

            # determine the range of alpha2 L and H      in page of 126
            # if y1 != y2    L = max(0, old_alpha2 - old_alpha1), H = min(C, C + old_alpha2 - old_alpha1)
            # if y1 == y2    L = max(0, old_alpha2 + old_alpha1 - C), H = min(C, old_alpha2 + old_alpha1)
            if self.train_label[i] != self.train_label[j]:
                L = max(0, old_alpha2 - old_alpha1)
                H = min(self.C, self.C + old_alpha2 - old_alpha1)
            else:
                L = max(0, old_alpha2 + old_alpha1 - self.C)
                H = min(self.C, old_alpha2 + old_alpha2)

            if L == H:
                # print("L == H")
                return 0

            # calculate eta in page of 127 Eq.(7.107)
            # eta = K11 + K22 - 2K12
            K11 = self.K[i, i]
            K12 = self.K[i, j]
            K21 = self.K[j, i]
            K22 = self.K[j, j]
            eta = K11 + K22 - 2 * K12
            if eta <= 0:
                # print("eta <= 0")
                return 0

            # update alpha2 and its error in page of 127 Eq.(7.106) and Eq.(7.108)
            self.alphas[j] = old_alpha2 + self.train_label[j]*(Ei - Ej)/eta
            self.alphas[j] = self.updateAlpha2(self.alphas[j], L, H)
            new_alphas2 = self.alphas[j]
            self.upadateError(j)

            # # if the stripe of alpha2 is not big enough, stop
            # if abs(self.alphas[j] - old_alpha2) < 0.01:
            #     return 0

            # update the alpha1 and its error in page of 127 Eq.(7.109)
            # new_alpha1 = old_alpha1 + y1y2(old_alpha2 - new_alpha2)
            new_alphas1 = old_alpha1 + self.train_label[i] * self.train_label[j] * (old_alpha2 - new_alphas2)
            self.alphas[i] = new_alphas1
            self.upadateError(i)

            # determine b in page of 130 Eq.(7.115) and Eq.(7.116)
            # new_b1 = -E1 - y1K11(new_alpha1 - old_alpha1) - y2K21(new_alpha2 - old_alpha2) + old_b
            # new_b2 = -E2 - y1K12(new_alpha1 - old_alpha1) - y2K22(new_alpha2 - old_alpha2) + old_b
            b1 = - Ei - self.train_label[i] * K11 * (old_alpha1 - self.alphas[i]) - self.train_label[j] * K21 * (old_alpha2 - self.alphas[j]) + self.b
            b2 = - Ej - self.train_label[i] * K12 * (old_alpha1 - self.alphas[i]) - self.train_label[j] * K22 * (old_alpha2 - self.alphas[j]) + self.b
            if (self.alphas[i] > 0) and (self.alphas[i] < self.C):
                self.b = b1
            elif (self.alphas[j] > 0) and (self.alphas[j] < self.C):
                self.b = b2
            else:
                self.b = (b1 + b2)/2.0

            return 1
        else:
            return 0

    '''
       Function:  SMO
       Description: implement of  Platt SMO, first search support vector which are not in bound, if alpha2 dosen't change enough, search 
                    the entire set 
       '''
    def SMO(self):
        iter = 0
        entire_set = True
        alpha_pairs_changes = 0
        while (iter < self.max_iteration) and (alpha_pairs_changes > 0) or entire_set:
            alpha_pairs_changes = 0
            if entire_set:
                for i in range(self.sample_num):
                    alpha_pairs_changes += self.innerLoop(i)
                    # print("Iteration:%d, Sample:%d, Pairs changed:%d" %(iter, i, alpha_pairs_changes))
                iter += 1
            else:
                non_bound_alpha = np.nonzero((self.alphas > 0) & (self.alphas < self.C))[0] # in page of 129 Eq.(7.112)
                for i in range(len(non_bound_alpha)):
                    alpha_pairs_changes += self.innerLoop(i)
                    # print("Iteration:%d, Sample:%d, Pairs changed:%d" % (iter, i, alpha_pairs_changes))
                iter += 1
            if entire_set:
                entire_set = False
            elif alpha_pairs_changes == 0:
                entire_set = True

        # print("Iteration:%d" % iter)

    '''
          Function:  kernel transformation
          Description: transform {0, 1} into {-1, 1}, list to ndarray
          Input: data           dataType: ndarray         description: data set
                 sample         dataType: ndarray         description: a sample
          Output: new_label     dataType: ndarray     description: new label
          '''

    def kernelTransformation(self, data, sample, kernel):
        sample_num, feature_dim = np.shape(data)
        K = np.zeros([sample_num])
        if kernel == "linear":  # linear function
            K = np.dot(data, sample.T)
        elif kernel == "poly":  # polynomial function
            K = (np.dot(data, sample.T) + self.c) ** self.n
        elif kernel == "sigmoid":
            K = np.tanh(self.g * np.dot(data, sample.T) + self.c)
        elif kernel == "rbf":  # Gaussian function
            for i in range(sample_num):
                delta = data[i, :] - sample
                K[i] = np.dot(delta, delta.T)
            K = np.exp(-self.g * K)
        else:
            raise NameError('Unrecognized kernel function')
        return K

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

        # initiation
        sample_num, feature_dim = np.shape(train_data)
        self.train_data = train_data
        self.train_label = self.labelTransformation(train_label)
        self.sample_num = sample_num
        self.K = np.zeros([self.sample_num, self.sample_num])
        self.alphas = np.zeros([self.sample_num, 1])
        self.errors = np.zeros([self.sample_num, 2])
        self.b = 0

        # kernel trick
        for i in range(self.sample_num):
            self.K[:, i] = self.kernelTransformation(self.train_data, self.train_data[i, :], self.kernel)

        # train model
        self.SMO()
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

        # find the support vectors and its corresponding label
        support_vectors_index = np.nonzero(self.alphas > 0)[0]
        support_vectors = self.train_data[support_vectors_index]
        support_vectors_label = self.train_label[support_vectors_index]
        support_vectors_alphas = self.alphas[support_vectors_index]

        # predict the test sample in page of 122 Eq.(7.89)
        for i in range(test_num):
            kernel_data = self.kernelTransformation(support_vectors, test_data[i, :], self.kernel)
            probability[i] = np.dot(kernel_data.T, np.multiply(support_vectors_label, support_vectors_alphas)) + self.b
            if probability[i] > 0:
                prediction[i] = 1
            else:
                prediction[i] = -1

        self.prediction = prediction
        self.probability = probability
        if prob:
            return probability
        else:
            return prediction

    '''
    Function:  accuracy
    Description: show detection result
    Input:  test_label dataType: ndarray   description: labels of test data
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
        model = {'b': self.b, 'alphas': self.alphas, 'labels': self.train_label}
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
        self.alphas = model['alphas']
        self.b = model['b']
        self.train_label = model['train_label']
        return self


