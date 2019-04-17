import numpy as np

'''
Function:  Normalization
Description: Normalize input data. For vector x, the normalization process is given by
            normalization(x) = (x - min(x))/(max(x) - min(x))
Input:  data        dataType: ndarray   description: input data
Output: normdata    dataType: ndarray   description: output data after normalization
'''

def Normalization(data):
    # get the max and min value of each column
    minValue = data.min(axis=0)
    maxValue = data.max(axis=0)
    diff = maxValue - minValue
    # normalization
    mindata = np.tile(minValue, (data.shape[0], 1))
    normdata = (data - mindata) / np.tile(diff, (data.shape[0], 1))
    return normdata

'''
Function:  Standardization
Description: Standardize input data. For vector x, the normalization process is given by
             Standardization(x) = x - mean(x)/std(x)
Input:  data            dataType: ndarray   description: input data
Output: standarddata    dataType: ndarray   description: output data after standardization
'''

def Standardization(data):
    # get the mean and the variance of each column
    meanValue = data.mean(axis=0)
    varValue = data.std(axis=0)
    standarddata = (data - np.tile(meanValue, (data.shape[0], 1))) / np.tile(varValue, (data.shape[0], 1))
    return standarddata
