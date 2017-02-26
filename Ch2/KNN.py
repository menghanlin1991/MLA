# File: KNN.py
# Name: mhl
# Email: xx

# import the library
import numpy as np
import operator
import matplotlib.pyplot as plt
from os import listdir

#Collect Data
'''
important, but null, already have
'''

#Prepare Data
def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    numOfLines = len(lines)
    train_x = np.zeros((numOfLines,3))
    train_y = []
    index_row = 0
    for loopLine in lines:
        loopLine = loopLine.strip()
        loopLine = loopLine.split('\t')
        train_x[index_row,:] = loopLine[0:3]
        index_row += 1
        train_y.append(int(loopLine[-1]))
    return train_x,train_y

def img2vector(filename):
    lines,numOfLine,lenLine = sizeImg(filename)
    returnVec = []
    #print(lenLine)
    for loopLine in lines:
        for loopStr in range(lenLine):
            returnVec.append(loopLine[loopStr])
    returnVecArray = np.array(returnVec)
    #print(returnVecArray.reshape(numOfLine,lenLine))
    return returnVec

def sizeImg(filename):
    # avoid not normal matrix
    fr = open(filename)
    lines = fr.readlines()
    numOfLine = len(lines)
    lineSample = lines[0].strip()
    lenLine = len(lineSample)
    return lines,numOfLine,lenLine

def prepareTrainData():
    train_y = []
    path = '/Users/mhl/Documents/MhlCode/mla/Ch2/digits/trainingDigits/'
    trainingList = listdir(path)
    numOfTrain = len(trainingList)
    lines,numOfLine,lenLine = sizeImg(path + trainingList[0])
    train_x = np.zeros(numOfTrain,numOfLine*lenLine)
    indexRow = 0
    for loopFile in trainingList:
        returnVec = img2vector(path+loopFile)
        train_x[indexRow,:] = returnVec
        indexRow += 1
        train_y.append(int(loopFile.split('_')[0]))
    return train_x,train_y

def prepareData(path):
    data_y = []
    dataList = listdir(path)
    print(dataList)
    del(dataList[0])
    numOfData = len(dataList)
    lines,numOfLine,lenLine = sizeImg(path + dataList[0])
    data_x = np.zeros((numOfData,numOfLine*lenLine))
    indexRow = 0
    for loopFile in dataList:
        returnVec = img2vector(path+loopFile)
        data_x[indexRow,:] = returnVec
        indexRow += 1
        data_y.append(int(loopFile.split('_')[0]))
    return data_x,data_y

#Analysis Data
def drawData(train_x,train_y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(train_x[:,1],train_x[:,2],15.0*np.array(train_y),15.0*np.array(train_y))
    plt.show()

#Norm Data
def autoNorm(dataSet):
    minCol = dataSet.min(0)
    maxCol = dataSet.max(0)
    diffMaxMin = maxCol - minCol
    numData = dataSet.shape[0]
    diffDataMin = dataSet - np.tile(minCol,(numData,1))
    diffMaxMinTemp = np.tile(diffMaxMin,(numData,1))
    normData = diffDataMin/diffMaxMinTemp
    return normData,diffMaxMin,minCol

#Def KnnClassify
def KnnClassify(test_x, train_x, train_y, k):
    train_size = train_x.shape[0]
    diff = np.tile(test_x, (train_size, 1)) - train_x
    square_diff = diff ** 2
    sum_diff = square_diff.sum(axis=1)
    diff_res = sum_diff ** 0.5
    indexDisSort = diff_res.argsort()
    print (indexDisSort)
    count = {}
    for loop in range(k):
        key = train_y[indexDisSort[loop]]
        count[key] = count.get(key, 0) + 1
    sortCount = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    return sortCount[0][0]

#Test Knn
def KnnClassifyTest(hoRatio):
    filename = 'datingTestSet.txt'
    train_x,train_y = file2matrix(filename)
    train_x_norm = autoNorm(train_x)
    numData = train_x_norm.shape[0]
    numDataTest = int(numData*hoRatio)
    errorCount = 0
    for loop in range(numDataTest):
        KnnRes = knnClassify(train_x_norm[loop,:], train_x_norm[numDataTest:numData,:],\
                             train_y[numDataTest:numData],3)
        if(KnnRes != train_y[loop]):
            errorCount += 1
            print "KnnClassify Output is: %d, the real y is: %d" % (KnnRes, train_y[loop])
    errorRatio = errorCount/float(numDataTest)
    print "error Ratio is %f" % (errorRatio)

def handWritingClassifyTest():
    pathTrain = '/Users/mhl/Documents/MhlCode/mla/Ch2/digits/trainingDigits/'
    train_x,train_y = prepareData(pathTrain)
    print(train_x)
    pathTest = '/Users/mhl/Documents/MhlCode/mla/Ch2/digits/testDigits/'
    test_x,test_y = prepareData(pathTest)
    print(test_x)
    sizeTest = test_x.shape[0]
    errorCount = 0
    for loopTest in range(sizeTest):
        res = KnnClassify(test_x[loopTest],train_x,train_y,3)
        if(res != test_y[loopTest]):
            errorCount += 1
            print 'the Knn out put is: %d, it should be: %d' % (res, test_y[loopTest])
    print 'error ratio is: %f' % (float(errorCount)/sizeTest)

#Apply Knn
def outputClassifyRes():
    classList = ['not all all', 'in small doses', 'in large doses']
    videoGame = float(raw_input("percentage of time cost on VideoGames every year?"))
    flier = float(raw_input("percentage of time cost on flier every year?"))
    iceCream = float(raw_input("percentage of time cost on iceCream every year?"))
    test_x = np.array([videoGame, flier, iceCream])
    #test_x = np.array([2,3,4])
    path = '/Users/mhl/Documents/MhlCode/mla/Ch2/'
    filename = path + 'datingTestSet2.txt'
    train_x, train_y = file2matrix(filename)
    #draw
    drawData(train_x,train_y)
    train_x_norm, diffMaxMin, minCol = autoNorm(train_x)
    test_x_norm = (test_x - minCol) / diffMaxMin
    print(test_x_norm)
    indexClassPredict = KnnClassify(test_x_norm, train_x_norm, train_y, 3)
    print "predict result is: %s" % (classList[indexClassPredict - 1])


#Main
#outputClassifyRes()
handWritingClassifyTest()
