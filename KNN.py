import numpy as np
import operator as op

class KNN:
    def __init__(self, k, normType):
        self.k = 10
        self.normType = "Normalization"

    '''
    Function:  Normalization
    Description: Normalize input data. For vector x, the normalization process is given by
                 normalization(x) = (x - min(x))/(max(x) - min(x))
    Input:  data        dataType: ndarray   description: input data
    Output: normdata    dataType: ndarray   description: output data after normalization
    '''
    def Normalization(self, data):
        # get the max and min value of each column
        minValue = data.min(axis=0)
        maxValue = data.max(axis=0)
        diff = maxValue - minValue
        # normalization
        mindata = np.tile(minValue, (data.shape[0], 1))
        normdata = (data - mindata)/np.tile(diff, (data.shape[0], 1))
        return normdata

    '''
    Function:  Standardization
    Description: Standardize input data. For vector x, the normalization process is given by
                 Standardization(x) = x - mean(x)/std(x)
    Input:  data            dataType: ndarray   description: input data
    Output: standarddata    dataType: ndarray   description: output data after standardization
    '''
    def Standardization(self, data):
        # get the mean and the variance of each column
        meanValue = data.mean(axis=0)
        varValue = data.std(axis=0)
        standarddata = (data - np.tile(meanValue, (data.shape[0], 1)))/np.tile(varValue, (data.shape[0], 1))
        return standarddata

    '''
    Function:  predict
    Description: give the prediction for test data
    Input:  testData    dataType: ndarray   description: data for testing
            trainData   dataType: ndarray   description: data for training
            trainLabel  dataType: ndarray   description: labels of train data
            k           dataType: int       description: select the first k distances
            normType    dataType: string    description: type of normalization, default:Normalization
            probability dataType: bool      description: if true return label and probability, else return label only
            showResult  dataType: bool      description: display the prediction result
    Output: results     dataType: ndarray   description: label or probability
    '''
    def predict(self, testData, trainData, trainLabel, k, normType="Normalization", probability=True, showResult=True):
        # Normalization
        if normType == "Standardization":
            trainData = KNN.Standardization(self, trainData)
            testData = KNN.Standardization(self, testData)

        test_num = testData.shape[0]
        results = np.zeros([test_num, 2])
        correct = 0
        # predict each samples in test data
        for i in range(test_num):
            results[i][0], results[i][1] = KNN.calcuateDistance(self, testData[i], trainData, trainLabel, k)
            if results[i][0] == trainLabel[i]:
                correct = correct + 1

        if showResult:
            print("The accuarcy is %f" % correct/test_num)

        if probability:
            return results
        else:
            return results[:0]

    '''
    Function:  calcuateDistance
    Description: calcuate the distance between input vector and train data
    Input:  input      dataType: ndarray   description: input vector
            trainData  dataType: ndarray   description: data for training
            trainLabel dataType: ndarray   description: labels of train data
            k          dataType: int       description: select the first k distances
    Output: pro        dataType: float     description: max probability of prediction 
            label      dataType: int       description: prediction label of input vector
    '''
    def calcuateDistance(self, input, trainData, trainLabel, k):
        train_num = trainData.shape[0]
        # calcuate the distances
        distances = np.tile(input, (train_num, 1)) - trainData
        distances = distances**2
        distances = distances.sum(axis=1)
        distances = distances**0.5

        # get the labels of the first k distances
        disIndex = distances.argsort()
        labelCount = {}
        for i in range(k):
            label = trainLabel[disIndex[i]]
            labelCount[label] = labelCount.get(label, 0) + 1

        prediction = sorted(labelCount.items(), key=op.itemgetter(1), reverse=True)
        label = prediction[0][0]
        pro = prediction[0][1]/k
        return label, pro


