import numpy as np
import math
import feedparser
import operator
from os import listdir

#Prepare data
#not normal one
'''
def loadDataSet():
    train_x = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    train_y = [0,1,0,1,0,1]
    return train_x,train_y
'''

def textParse(inputString):
    import re
    listTokens = re.split(r'\W*',inputString)
    listTokens = [loop.lower() for loop in listTokens if len(loop)>2]
    return listTokens

#not cool method
'''
def createVocaList(train_x):
    tempDict = {}
    for loop in train_x:
        for index in loop:
            if index not in tempDict.keys():
                tempDict[index] = 0
            tempDict[index] += 1
    vocaList = list(set(tempDict))
    return vocaList
'''

def createVocaList(train_x):
    vocaList = []
    for loop in train_x:
        vocaList.extend(loop)
    tempSet = set(vocaList)
    vocaList = list(tempSet)
    return vocaList

#another method
'''
def train2vec(train_x, vocaList):
    numVoca = len(vocaList)
    numTrain_x = len(train_x)
    outVec = np.zeros((numTrain_x,numVoca))
    cmpDict = {}
    index = 0
    for loop in vocaList:
        cmpDict[loop] = index
        index += 1
    index_below = 0
    for loopRow in train_x:
        for loopCol in loopRow:
            if loopCol in cmpDict.keys():
                outVec[index_below][cmpDict[loopCol]] = 1
        index_below += 1
    return outVec
'''
def changeVocaList(vocaList,fullText,delNum):
    vocaDict = {}
    for loop in vocaList:
        vocaDict[loop] = fullText.count(loop)
    vocaSorted = sorted(vocaDict.iteritems(),key=operator.itemgetter(1),reverse=True)
    vocaDel = vocaSorted[0:delNum]
    for loopDel in range(len(vocaDel)):
        del(vocaList[vocaList.index(vocaDel[loopDel][0])])
    return vocaList


def input2vec(input_x, vocaList):
    numVoca = len(vocaList)
    numInput_x = len(input_x)
    outVec = np.zeros((numInput_x,numVoca))
    index = 0
    for loopRow in input_x:
        for loopCol in loopRow:
            if loopCol in vocaList:
                outVec[index][vocaList.index(loopCol)] += 1
        index += 1
    return outVec

def loadDataSet(path):
    labelClass = listdir(path)
    del(labelClass[0])
    numClass = len(labelClass)
    train_x = []
    train_y = []
    fullText = []
    for loop in range(numClass):
        mailList = listdir(path+labelClass[loop])
        del(mailList[0])
        numList = len(mailList)
        for loopMail in mailList:
            strMail = open(path+labelClass[loop]+'/%s' % loopMail).read()
            train_x.append(textParse(strMail))
            fullText.append(textParse(strMail))
            train_y.append(loop)
    vocaListTemp = createVocaList(train_x)
    #print(len(vocaListTemp))
    vocaList = changeVocaList(vocaListTemp,fullText,0)
    #print(len(vocaList))
    train_x = input2vec(train_x,vocaList)
    train_y = np.array(train_y)
    #print(train_x,train_y)
    #print (train_x.shape)
    return train_x,train_y,vocaList

#Get data from web and save
def getDataSet():
    #should add more here, if add here, the loadDataSetAd will need feed0 and feed1 as input args
    print('NULL')

#Prepare data
def loadDataSetAd():
    totalClass = [feedparser.parse('http://newyork.craigslist.org/stp/index.rss'),\
             feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')]
    numClass = len(totalClass)
    train_x = []
    train_y = []
    fullText = []
    minLen = min(len(totalClass[loop]['entries']) for loop in range(numClass))
    #print(minLen)
    for loopClass in range(numClass):
        for loopStr in range(minLen):
            train_x.append(textParse(totalClass[loopClass]['entries'][loopStr]['summary']))
            fullText.extend(textParse(totalClass[loopClass]['entries'][loopStr]['summary']))
            train_y.append(loopClass)
    vocaListTemp = createVocaList(train_x)
    vocaList = changeVocaList(vocaListTemp,fullText,5)
    print(vocaList)
    train_x = input2vec(train_x,vocaList)
    train_y = np.array(train_y)
    #print(train_x,train_y)
    return train_x,train_y,vocaList
    #for loop in range(numClass):

#sleclt 20% as test_x and test_y
def partitionTestSet(train_x,train_y):
    numTest = int(train_x.shape[0] * 0.2)
    numTestCol = len(train_x[0])
    test_x = np.zeros((numTest,numTestCol))
    test_y = np.zeros(numTest)
    train_x = train_x.tolist()
    train_y = train_y.tolist()
    for loop in range(numTest):
        randomIndex = int(np.random.uniform(0,len(train_x)))
        test_x[loop] = train_x[randomIndex]
        test_y[loop] = train_y[randomIndex]
        #train_x.remove(train_x[randomIndex])
        del(train_x[randomIndex])
        #train_y.remove(train_y[randomIndex])
        del(train_y[randomIndex])
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    #print(train_x,train_y,test_x,test_y)
    return train_x,train_y,test_x,test_y

#Anylysize Data
def getFreqTopWords(pWordEachClass,vocaList):
    sizeCol = len(pWordEachClass[pWordEachClass.keys()[0]])
    #print(len(vocaList),sizeCol,pWordEachClass)
    classTopWords = []
    threshold = -6.0
    topWordsSorted = []
    index = 0
    labels = []
    for loopClass in pWordEachClass.keys():
        classTopWords.append([(vocaList[loopIndex],pWordEachClass[loopClass][loopIndex]) \
            for loopIndex in range(sizeCol) \
            if pWordEachClass[loopClass][loopIndex] > threshold])
        #print(classTopWords)
        tempSorted = sorted(classTopWords[index],key=lambda pair: pair[1],reverse=True)
        topWordsSorted.append(tempSorted)
        index += 1
        tempLabel = [tempSorted[loopIndex][0] for loopIndex in range(len(tempSorted))]
        labels.append(tempLabel)
        print('the top words of class%d is:' % loopClass,labels[loopClass])


    #lack of sort

#Train
def bayesClassify(train_x, train_y):
    #get the class num
    numClass = {}
    for loop in train_y:
        if loop not in numClass.keys():
            numClass[loop] = 0
        numClass[loop] += 1
    #print(numClass.keys())
    #calculate the probability of each class
    numTrain = len(train_y)
    pClass = {}
    for loop in numClass.keys():
        pClass[loop] = math.log(float(numClass[loop])/numTrain)
    #sizeNumClass = numClass.__len__()
    #print(sizeNumClass)
    #print(pClass)
    #calculate each word probability under different class
    pWordEachClass = {}
    pWord = {}
    #make sure the Dict has initial value
    for loopClass in numClass.keys():
        pWordEachClass[loopClass] = np.ones(len(train_x[0]))
        pWord[loopClass] = 2
    indexTrain = 0
    for index in train_y:
        for loopClass in numClass.keys():
            if index == loopClass:
                pWordEachClass[loopClass] += train_x[indexTrain]
                pWord[loopClass] += sum(train_x[indexTrain])
        indexTrain += 1
    for loopClass in numClass.keys():
        pWordEachClass[loopClass] /= pWord[loopClass]
        pWordEachClass[loopClass] = np.array([math.log(loop) for loop in pWordEachClass[loopClass]])
    #print(pClass, pWordEachClass)
    return pClass,pWordEachClass

#Test Classify
def bayesTestBasic(test_x,pClass,pWordEachClass):
    #transform test_x to vector
    #test_x = input2vec(test_x,vocaList)  already done before, so delete here
    #need return test_y
    numTest = len(test_x)
    index = 0
    test_y = np.zeros(numTest)
    for loop in test_x:
        pMax = -100000000
        classMax = pClass.keys()[0]
        for loopClass in pClass.keys():
            pTemp = (sum(pWordEachClass[loopClass] * loop) + pClass[loopClass])
            if pTemp > pMax:
                pMax = pTemp
                classMax = loopClass
        test_y[index] = classMax
        index += 1
    #print(test_y)
    return test_y

def bayesCrossVal(test_x,test_y,pClass,pWordEachClass):
    test_y_cal = bayesTestBasic(test_x,pClass,pWordEachClass)
    numTest = test_y.shape[0]
    errorCount = 0
    for loop in range(numTest):
        if test_y_cal[loop] != test_y[loop]:
            errorCount += 1
    print(test_y_cal,test_y)
    print(float(errorCount)/numTest)
    return errorCount

'''
train_x,train_y = loadDataSet()
vocaList = createVocaList(train_x)
train_x = input2vec(train_x,vocaList)
pClass,pWordEachClass = bayesClassify(train_x,train_y)
test_x = [['love','my','dalmation'],['stupid','garbage']]
bayesTestBasic(test_x,pClass,pWordEachClass)
'''
'''
path = '/Users/mhl/Documents/MhlCode/mla/Ch4/email/'
#Prepare Data
train_x, train_y,vocaList = loadDataSet(path)
train_x,train_y,test_x,test_y = partitionTestSet(train_x,train_y)
#Train Data
pClass,pWordEachClass = bayesClassify(train_x,train_y)
#Anylysize
getFreqTopWords(pWordEachClass,vocaList)
#Test Data
bayesCrossVal(test_x,test_y,pClass,pWordEachClass)
'''
#'''
#Prepare Data
train_x,train_y,vocaList = loadDataSetAd()
train_x,train_y,test_x,test_y = partitionTestSet(train_x,train_y)
#Train Data
pClass,pWordEachClass = bayesClassify(train_x,train_y)
#Anylysize
getFreqTopWords(pWordEachClass,vocaList)
#Test Data
bayesCrossVal(test_x,test_y,pClass,pWordEachClass)
#'''