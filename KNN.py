import numpy as np
import operator as op

class KNNClassifier:
    def __init__(self, k, normType="Normalization"):
        self.k = k
        self.normType = "Normalization"
        self.x_train = None
        self.y_train = None

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
    Function:  train
    Description: train the model
    Input:  trainData       dataType: ndarray   description: features
            testData        dataType: ndarray   description: labels
    Output: self            dataType: obj       description: 
    '''
    def train(self,trainData, trainLabel):
        if self.normType == "Standardization":
            trainData = self.Standardization(trainData)
        else:
            trainData = self.Normalization(trainData)
        self.x_train = trainData
        self.y_train = trainLabel
        return self

    '''
    Function:  predict
    Description: give the prediction for test data
    Input:  testData    dataType: ndarray   description: data for testing
            testLabel  dataType: ndarray   description: labels of train data
            normType    dataType: string    description: type of normalization, default:Normalization
            probability dataType: bool      description: if true return label and probability, else return label only
            showResult  dataType: bool      description: display the prediction result
    Output: results     dataType: ndarray   description: label or probability
    '''
    def predict(self, testData):
        # Normalization
        if self.normType == "Standardization":
            testData = self.Standardization(testData)
        else:
            testData = self.Normalization(testData)

        test_num = testData.shape[0]
        prediction = np.zeros([test_num, 1])
        probability = np.zeros([test_num, 1])
        # predict each samples in test data
        for i in range(test_num):
            prediction[i], probability[i] = self.calcuateDistance(testData[i], self.x_train, self.y_train, self.k)

        return prediction

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

    '''
    Function:  showDetectionResult
    Description: show detection result
    Input:  testData  dataType: ndarray   description: data for test
            testLabel dataType: ndarray   description: labels of test data
    Output: accuracy  dataType: float     description: detection accuarcy
    '''
    def showDetectionResult(self, testData, testLabel):
        testLabel = np.expand_dims(testLabel,axis=1)
        prediction = self.predict(testData)
        accuarcy = sum(prediction == testLabel)/len(testLabel)
        return accuarcy
