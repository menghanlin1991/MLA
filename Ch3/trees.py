from math import log
import operator
import matplotlib.pyplot as plt

def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    dataLabels = ['not surfacing','flippers']
    return dataSet,dataLabels

def calShannonEnt(dataSet):
    classCount = {}
    numOfSample = len(dataSet)
    shannonEntExp = 0.0
    for loopSample in dataSet:
        classKey = loopSample[-1]
        if classKey not in classCount.keys():
            classCount[classKey] = 0
        classCount[classKey] += 1
    for key in classCount:
        prob = float(classCount[key])/numOfSample
        shannonEntExp -= prob*log(prob,2)
    #print(shannonEntExp)
    return shannonEntExp

def conditionDataSet(dataSet,feature,value):
    retDataSet = []
    for loopSample in dataSet:
        #print(loopSample)
        if loopSample[feature] == value:
            delSample = loopSample[:feature]
            #print(delSample)
            delSample.extend(loopSample[feature+1:])
            retDataSet.append(delSample)
    return retDataSet

def ChooseBestFeature(dataSet):
    numFeature = len(dataSet[0])-1
    baseEnt = calShannonEnt(dataSet)
    maxInfGain = 0.0
    for loopFeature in range(numFeature):
        newEnt = 0.0
        loopFeatureValue = [i[loopFeature] for i in dataSet]
        loopFeatureUniqueValue = set(loopFeatureValue)
        for index in loopFeatureUniqueValue:
            retDataSet = conditionDataSet(dataSet,loopFeature,index)
            prob = float(len(retDataSet))/len(dataSet)
            #print(calShannonEnt(retDataSet))
            newEnt += prob*calShannonEnt(retDataSet)
        tempInfGain = baseEnt-newEnt
        if tempInfGain > maxInfGain:
            maxInfGain = tempInfGain
            bestFeature = loopFeature
    return bestFeature

def majorityClass(classList):
    classCount = {}
    #sizeClassList = len(classList)
    for loopClass in classList:
        if loopClass not in classCount.keys():
            classCount[loopClass] = 0
        classCount[loopClass] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    #print(dataSet)
    classList = [loop[-1] for loop in dataSet]
    # the first stop condition
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #the second stop condition
    if len(dataSet[0]) == 1:
        print 'haha'
        return majorityClass(classList)
    #just a int data
    bestFeature = ChooseBestFeature(dataSet)
    #change to label
    #print(bestFeature)
    bestFeatureLabel = labels[bestFeature]
    del(labels[bestFeature])
    myTree = {bestFeatureLabel:{}}
    featureVal = [loop[bestFeature] for loop in dataSet]
    FeatureValUnique = set(featureVal)
    for loopVal in FeatureValUnique:
        subLabels = labels[:]
        myTree[bestFeatureLabel][loopVal] = createTree(conditionDataSet\
                                                       (dataSet,bestFeature,loopVal),subLabels)
    return myTree

#Display Data
decisionNode = dict(boxstyle='sawtooth',fc='0.8')
leafNode = dict(boxstyle='round4',fc='0.8')
arrow_args = dict(arrowstyle='<-')

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',\
                            xytext=centerPt,textcoords='axes fraction',\
                            va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)
#just for test
'''
def createPlot():
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=True)
    plotNode('decision node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
'''
def getLeafsNum(inputTree):
    leafNum = 0
    head = inputTree.keys()[0]
    other = inputTree[head]
    for loopKey in other.keys():
        if type(other[loopKey]).__name__ != 'dict':
            leafNum += 1
        else:
            leafNum += getLeafsNum(other[loopKey])
    return leafNum

def getTreeDepth(inputTree):
    maxDepth = 0
    head = inputTree.keys()[0]
    other = inputTree[head]
    for loopKey in other.keys():
        if type(other[loopKey]).__name__ != 'dict':
            depth = 1
        else:
            depth = getTreeDepth(other[loopKey]) + 1
        #print 'depth: %d' % depth
        if depth > maxDepth:
            maxDepth = depth
        #print 'maxDepth: %d' % maxDepth
    return maxDepth

def treeClassifyTest(inputTree, testVec):
    head = inputTree.keys()[0]
    other = inputTree[head]
    valCmp = testVec[0]
    for loopKey in other.keys():
        if valCmp == loopKey:
            if type(other[loopKey]).__name__ == 'dict':
                del(testVec[0])
                return treeClassifyTest(other[loopKey],testVec)
            else:
                return other[loopKey]

def saveTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def openTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

def createTreeGlass(filename):
    fr = open(filename)
    lines = fr.readlines()
    dataSet = []
    for loopLine in lines:
        loopLine = loopLine.strip()
        dataSet.append(loopLine.split('\t'))
    print(dataSet)
    labels = ['age','prescript','astigmatic','tearRate']
    #print(createTree(dataSet,labels))
    return createTree(dataSet,labels)

def plotMidText(currentPt, parentPt, txtString):
    xMid = (currentPt[0]-parentPt[0])/2.0+parentPt[0]
    yMid = (parentPt[1]-currentPt[1])/2.0+currentPt[1]
    createPlot.ax1.text(xMid,yMid,txtString)

def plotTree(inputTree,parentPT,nodeTxt):
    numLeafs = getLeafsNum(inputTree)
    head = inputTree.keys()[0]
    currentPt = (plotTree.xOff+(1+float(numLeafs))/2/plotTree.totalW,plotTree.yOff)
    plotMidText(currentPt,parentPT,nodeTxt)
    plotNode(head,currentPt,parentPT,decisionNode)
    other = inputTree[head]
    plotTree.yOff -= 1.0/plotTree.totalD
    for loop in other.keys():
        if type(other[loop]).__name__ == 'dict':
            plotTree(other[loop],currentPt,str(loop))
        else:
            plotTree.xOff += 1.0/plotTree.totalW
            plotNode(other[loop],(plotTree.xOff,plotTree.yOff),currentPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),currentPt,str(loop))
    plotTree.yOff += 1.0/plotTree.totalD

def createPlot(inputTree):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = float(getLeafsNum(inputTree))
    plotTree.totalD = float(getTreeDepth(inputTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inputTree,(0.5,1.0),'')
    plt.show()

#main
#*********************debug**************************
#dataSet,dataLabels = createDataSet()
#print(dataSet)
#retDataSet = conditionDataSet(dataSet,0,1)
#print(retDataSet)
#calShannonEnt()
#print(ChooseBestFeature(dataSet))
#classList = ['a','b','c','a','b','b']
#print(majorityClass(classList))
#tree = {}
#tree = createTree(dataSet,dataLabels)
#testVec = [0,1]
#print(treeClassifyTest(tree,testVec))
#print(tree)
#print(getLeafsNum(tree))
#print(getTreeDepth(tree))
#createPlot()
#tree = createTree(dataSet,dataLabels)
#getLeafsNum(tree)
#saveTree(tree,'Tree.txt')
#print(openTree('Tree.txt'))
#***********************debug**************************

def main():
    path = '/Users/mhl/Documents/MhlCode/mla/Ch3/'
    filename = path + 'lenses.txt'
    inputTree = createTreeGlass(filename)
    print(inputTree)
    createPlot(inputTree)


if __name__ == "__main__":
    main()