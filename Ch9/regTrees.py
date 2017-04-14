import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import  Figure
import matplotlib.pyplot as plt
from Tkinter import *



class treeNode():
    def __init__(self, feat, val, left, right):
        self.sp_index = feat
        self.sp_val = val
        self.l_node = left
        self.r_node = right

def loadDataSet(filename):
    dataList = []
    fr = open(filename)
    for i in fr.readlines():
        temp = i.strip().split('\t')
        dataList.append(map(float, temp))
    dataMat = np.mat(dataList)
    #print (dataMat)
    return dataMat

def splitDataSet(dataMat, index, val):
    sp_mat1 = dataMat[np.nonzero(dataMat[:, index]<=val)[0], :][0]
    sp_mat2 = dataMat[np.nonzero(dataMat[:, index]>val)[0], :][0]
    #print (sp_mat1.shape, sp_mat2.shape)
    return sp_mat1, sp_mat2

def regLeaf(dataMat):
    return np.mean(dataMat[:, -1])

def regErr(dataMat):
    return np.var(dataMat[:, -1])*dataMat.shape[0]

def modelLeaf(data_mat):
    w = lineReg(data_mat)
    return w

def modelErr(data_mat):
    m,n = data_mat.shape
    x_mat = np.mat(np.ones((m, n)))
    x_mat[:, 1:n] = data_mat[:, :-1]
    y_mat = data_mat[:, -1]
    w = lineReg(data_mat)
    py = x_mat * w
    diff = py - y_mat
    #print ('diff', diff.T * diff)
    return (diff.T * diff)

def lineReg(data_mat):
    m,n = data_mat.shape
    x_mat = np.mat(np.ones((m, n)))
    x_mat[:, 1:n] = data_mat[:, :-1]
    y_mat = data_mat[:, -1]
    xTx = x_mat.T*x_mat
    #print (xTx)
    if (np.linalg.det(xTx) == 0.0):
        print 'wrong, the det is zero, can not calculate I'
        return 1
    w = xTx.I*x_mat.T*y_mat
    return w

def BestSplit(dataMat, leafType, errType, ops):
    #return if all vals are same
    if (len(set(dataMat[:, -1].T.tolist()[0])) == 1):
        return None, dataMat[0, -1]
    m, n = dataMat.shape
    if (m == 1):
        return None, dataMat[-1]
    n = n-1 #delete y
    min_err = np.Inf
    pre_err = errType(dataMat)
    sp_index = 0
    sp_val = 0
    for i in range(n):
        for j in set(dataMat[:, i]):
            mat1, mat2= splitDataSet(dataMat, i, j)
            #if the set is small, return leafNode
            if ((mat1.shape[0] < ops[1]) or (mat2.shape[0] < ops[1])):
                continue
            mat1_err = errType(mat1)
            mat2_err = errType(mat2)
            sp_err = mat1_err + mat2_err
            if (sp_err < min_err):
                sp_index = i
                sp_val = j
                min_err = sp_err
                #print(sp_index, sp_val)
    #if dif is small, return leaf
    if (pre_err - min_err < ops[0]):
        return None, leafType(dataMat)
    mat1, mat2 = splitDataSet(dataMat, sp_index, sp_val)
    #if set is small, return leaf
    if ((mat1.shape[0] < ops[1]) or (mat2.shape[0] < ops[1])):
        return None, leafType(dataMat)
    return sp_index, sp_val

def createSplitTree(dataMat, leafType=regLeaf, errType=regErr, ops=(1, 20)):
    #if leafNode, will return leafNode
    sp_index, sp_val = BestSplit(dataMat, leafType, errType, ops)
    #print (sp_index, sp_val)
    if sp_index == None:
        return sp_val
    retTree = {}
    retTree['sp_index'] = sp_index
    retTree['sp_val'] = sp_val
    '''
    l_dataMat, r_dataMat = splitDataSet(dataMat, sp_index, sp_val)
    l_node = createSplitTree(l_dataMat)
    r_node = createSplitTree(r_dataMat)
    ret_node = treeNode(sp_index, sp_val, l_node, r_node)
    '''
    l_dataMat, r_dataMat = splitDataSet(dataMat, sp_index, sp_val)
    l_node = createSplitTree(l_dataMat, leafType, errType, ops)
    r_node = createSplitTree(r_dataMat, leafType, errType, ops)
    retTree['l_node'] = l_node
    retTree['r_node'] = r_node
    return retTree

def plot(dataMat, sp_val=1, l_node=1, r_node=1):
    train_x = dataMat[:, 0].T.tolist()[0]
    train_y = dataMat[:, 1].T.tolist()[0]
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.scatter(train_x, train_y)
    ax.plot([sp_val, sp_val], [-1, 2], c='b')
    ax.plot([-0.2, sp_val], [l_node, l_node], c='r')
    ax.plot([sp_val, 1.2], [r_node, r_node], c='r')
    plt.show()

def isTree(input_node):
    if (type(input_node).__name__=='dict'):
        return True

#this is from book, not necessary, only tree with lead node should compute mean
'''
def mean(input_tree):
    if isTree(input_tree['l_node']):
        input_tree['l_node'] = mean(input_tree['l_node'])
    if isTree(input_tree['r_node']):
        input_tree['r_node'] = mean(input_tree['r_node'])
    return ((input_tree['l_node']+input_tree['r_node'])/2.0)
'''

def mean(input_tree):
    return ((input_tree['l_node'] + input_tree['r_node']) / 2.0)

def prune(input_tree, test_data):
    if test_data.shape[0] == 0:
        return input_tree
    if (isTree(input_tree['l_node']) or isTree(input_tree['r_node'])):
        l_mat, r_mat = splitDataSet(test_data, input_tree['sp_index'], input_tree['sp_val'])
    if (isTree(input_tree['l_node'])):
        input_tree['l_node'] = prune(input_tree['l_node'], l_mat)
    if (isTree(input_tree['r_node'])):
        input_tree['r_node'] = prune(input_tree['r_node'], r_mat)
    if (not isTree(input_tree['l_node']) and not isTree(input_tree['r_node'])):
        l_mat, r_mat = splitDataSet(test_data, input_tree['sp_index'], input_tree['sp_val'])
        #cal left diff
        l_diff_mat = l_mat[:, -1] - input_tree['l_node']
        l_diff_val = l_diff_mat.T * l_diff_mat
        #cal right diff
        r_diff_mat = r_mat[:, -1] - input_tree['r_node']
        r_diff_val = r_diff_mat.T * r_diff_mat
        diff_total = l_diff_val + r_diff_val
        #cal mean
        mean_prune = mean(input_tree)
        mean_diff_mat = test_data[:, -1] - mean_prune
        mean_diff_val = mean_diff_mat.T * mean_diff_mat
        if (mean_diff_val < diff_total):
            print 'merging is better'
            return mean_prune
        else:
            return input_tree
    else:
        return input_tree

#def model type, regression and model
#test item is array
def regTreePre(leaf, test_item_arr):
    return float(leaf)

def modelTreePre(leaf, test_item_arr):
    n = test_item_arr.shape[0]
    x_mat = np.mat(np.ones((1, n)))
    x_mat[0, 1:n] = test_item_arr[:-1]
    return float(x_mat * leaf)

def recurPre(input_tree, test_item_arr, model_type):
    if not isTree(input_tree):
        return model_type(input_tree, test_item_arr)
    if (test_item_arr[input_tree['sp_index']] <= float(input_tree['sp_val'])):
        return recurPre(input_tree['l_node'], test_item_arr, model_type)
    else:
        return recurPre(input_tree['r_node'], test_item_arr, model_type)

def testPre(input_tree, test_mat, model_type=regTreePre):
    m, n = test_mat.shape
    py = []
    for i in range(m):
        py.append(recurPre(input_tree, test_mat[i].A[0], model_type))
    py = np.mat(py).T
    return py

def getInputs():
    try: tolS = float(tolSEntry.get())
    except:
        tolS = 1.0
        print 'enter float'
        tolSEntry.delete(0, END)
        tolSEntry.insert(0, '1.0')
    try: tolN = int(tolNEntry.get())
    except:
        tolN = 4
        print 'enter integer'
        tolSEntry.delete(0, END)
        tolSEntry.insert(0, '10')
    ops = (tolS, tolN)
    return ops

def reDraw():
    reDraw.f.clf()
    reDraw.ax = reDraw.f.add_subplot(111)
    ops_in = getInputs()
    if chkBtnVar.get():
        if (ops_in[1] < 2):
            ops_in[1] = 2
        ret_tree = createSplitTree(reDraw.data_mat, leafType=modelLeaf, errType=modelErr, ops=ops_in)
        py = testPre(ret_tree, reDraw.test_mat, model_type=modelTreePre)
    else:
        ret_tree = createSplitTree(reDraw.data_mat, leafType=regLeaf, errType=regErr, ops=ops_in)
        py = testPre(ret_tree, reDraw.test_mat, model_type=regTreePre)
    reDraw.ax.scatter(reDraw.data_mat[:, 0].T.A[0], reDraw.data_mat[:, 1].T.A[0], s=5)
    reDraw.ax.plot(reDraw.test_mat[:, 0].T.A[0], py.T.A[0])
    reDraw.canvas.show()
    #pass

path = '/Users/mhl/Documents/MhlCode/mla/Ch9/'
filename = path+'bikeSpeedVsIq_train.txt'
data_mat = loadDataSet(filename)
test_filename = path+'bikeSpeedVsIq_test.txt'
test_mat = loadDataSet(test_filename)

'''
#***********************
#splitDataSet(dataMat, 1, 0)
ret_tree = createSplitTree(data_mat, leafType=regLeaf, errType=regErr)
#plot(data_mat, retTree['sp_val'].A[0], retTree['l_node'], retTree['r_node'])
#plot(data_mat)
print(ret_tree)
#prune_tree = prune(retTree, test_mat)
#print(prune_tree)

py = testPre(ret_tree, test_mat, model_type=regTreePre)
print (py)
print (np.corrcoef(py, test_mat[:, -1], rowvar=0)[0,1])
#**************************

ret_tree = createSplitTree(data_mat, leafType=modelLeaf, errType=modelErr)
py = testPre(ret_tree, test_mat, model_type=modelTreePre)
print (py)
print (np.corrcoef(py, test_mat[:, -1], rowvar=0)[0,1])

#***************************
w = lineReg(data_mat)
m, n = test_mat.shape
x_mat = np.mat(np.zeros((m, n)))
x_mat[:, 1:n] = test_mat[:, :-1]
py = x_mat*w
print(py)
print (np.corrcoef(py, test_mat[:, -1], rowvar=0)[0,1])
'''

filename = path+'sine.txt'
#*****************************************************
root = Tk()
reDraw.f = Figure(figsize=(5, 4), dpi=100)
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)
#*****************************************************
#Label(root, text='Plot Place Holder').grid(row=0, columnspan=3)
Label(root, text='tolS').grid(row=1, column=0)
tolSEntry = Entry(root)
tolSEntry.grid(row=1, column=1)
tolSEntry.insert(0, '1.0')
Label(root, text='tolN').grid(row=2, column=0)
tolNEntry = Entry(root)
tolNEntry.grid(row=2, column=1)
tolNEntry.insert(0, '10')
Button(root, text='reDraw', command=reDraw).grid(row=1, column=2, rowspan=2)
chkBtnVar = IntVar()
chkBtn = Checkbutton(root, text='Model Tree', variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=3)
reDraw.data_mat = loadDataSet(filename)
#print (reDraw.data_mat)
temp = np.mat(np.arange(min(reDraw.data_mat[:, 0]), max(reDraw.data_mat[:, 0]), 0.01)).T
m, n = temp.shape
reDraw.test_mat = np.mat(np.ones((m, n+1)))
reDraw.test_mat[:, 0] = temp
reDraw()
root.mainloop()
