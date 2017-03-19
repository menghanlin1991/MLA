#from this py file, will use xx_xx_xx as variable format
import numpy as np
import matplotlib.pyplot as plt
import random

#Prepare Data
def loadDataSet(filename):
    fr = open(filename)
    lines = fr.readlines()
    len_row = len(lines)
    len_col = len(lines[0].strip().split())
    train_x = np.zeros((len_row,len_col))
    train_y = []
    index_row = 0
    for loop_line in lines:
        loop_line = loop_line.strip().split()
        train_x[index_row,0] = 1.0
        train_x[index_row,1:] = loop_line[:-1]
        train_y.append(int(float(loop_line[-1])))
        index_row += 1
    train_y = np.array(train_y)
    #print(train_x,train_y)
    return train_x,train_y

#Train
def sigmoid(input_x):
    return 1.0/(1+np.exp(-input_x))

def gradAscent(train_x,train_y,iter_num=500):
    train_x = np.mat(train_x)
    train_y = np.mat(train_y).transpose()
    iter_num = iter_num
    alpha = 0.001
    w = np.ones(train_x[0].shape)
    w = np.mat(w).transpose()
    for loop in range(iter_num):
        h = sigmoid(train_x*w)
        error = train_y - h
        w = w + alpha*train_x.transpose()*error
    #print(w)
    error = np.array(error)
    w = np.array(w)
    #analyseError(error)
    return w

def onlineGradAscent(train_x,train_y,iter_num=20):
    train_x = np.mat(train_x)
    train_y = np.mat(train_y).transpose()
    iter_num = iter_num
    alpha = 0.01
    num_row,num_col = train_x.shape
    w = np.ones(num_col)
    w = np.mat(w).transpose()
    for loop_iter in range(iter_num):
        error_record = []
        for i in range(num_row):
            h = sigmoid(train_x[i] * w)
            error = train_y[i] - h
            error_record.append(float(error))
            w = w + alpha * train_x[i].transpose() * error
    #print(w)
    error_record = np.array(error_record)
    w = np.array(w)
    #analyseError(error_record)
    return w

def onlineGradAscentV1(train_x,train_y,iter_num=20):
    train_x = np.mat(train_x)
    train_y = np.mat(train_y).transpose()
    iter_num = iter_num
    num_row,num_col = train_x.shape
    w = np.ones(num_col)
    w = np.mat(w).transpose()
    for loop_iter in range(iter_num):
        error_record = []
        dataIndex = range(num_row)
        for i in range(num_row):
            alpha = 4/(1.0+loop_iter+i) + 0.01
            data_index_pre = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(train_x[data_index_pre] * w)
            error = train_y[data_index_pre] - h
            error_record.append(float(error))
            w = w + alpha * train_x[data_index_pre].transpose() * error
    #print(w)
    error_record = np.array(error_record)
    w = np.array(w)
    #analyseError(error_record)
    return w

#Anylysis Error
def analyseError(error):
    coord_x = range(len(error))
    print(coord_x,error)
    plt.figure(1)
    ax1 = plt.subplot(111)
    plt.sca(ax1)
    plt.plot(coord_x,error,'r')
    plt.xlabel('Sample')
    plt.ylabel('Error')
    plt.title('Error of Each Sample')
    plt.xlim(0,len(error))
    plt.ylim(-1,1)
    plt.show()

#Display Data Classify
def plotBestFit(train_x,train_y,w):
    loop_size = len(train_y)
    x0 = []; y0 = []
    x1 = []; y1 = []
    for loop_index in range(loop_size):
        if train_y[loop_index] == 0:
            x0.append(train_x[loop_index][1])
            y0.append(train_x[loop_index][2])
        else:
            x1.append(train_x[loop_index][1])
            y1.append(train_x[loop_index][2])
    print(x0)
    print(y0)
    fig = plt.figure(1,facecolor='w')
    ax = fig.add_subplot(111)
    ax.scatter(x0,y0,c='r',marker='s')
    ax.scatter(x1,y1,c='g')
    x_line = np.arange(-3.0,3.0,0.2)
    y_line = (-w[0]-w[1]*x_line)/w[2]
    ax.plot(x_line,y_line)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

#Test Data
def sigmoidPred(input_x,w):
    input_x = np.mat(input_x)
    z = np.array(input_x*w)
    #print('z',z)
    output = sigmoid(z)
    #print('output',output)
    if output > 0.5:
        return 1
    else:
        return 0

def test(w,test_x,test_y):
    out_predict = []
    for input_x in test_x:
        out_predict.append(sigmoidPred(input_x,w))
    index = 0
    error_count = 0
    for i in range(len(test_y)):
        if out_predict[index] != test_y[index]:
            error_count += 1
        index += 1
    error_rate = float(error_count)/len(test_y)
    print('error_rate of this test is %f' % error_rate)
    return error_rate

def multiTest(test_x,test_y):
    num_test = 10
    error_rate = 0
    for loop in range(num_test):
        w = onlineGradAscentV1(train_x,train_y,100)
        error_rate += test(w,test_x,test_y)
    error_rate /= num_test
    print('error_rate of %d test is %f' % (num_test,error_rate))

#Debug
path = '/Users/mhl/Documents/MhlCode/mla/Ch5/'
filename = path+'horseColicTraining.txt'
train_x,train_y = loadDataSet(filename)
filenameTest = path+'horseColicTest.txt'
test_x,test_y = loadDataSet(filenameTest)
multiTest(test_x,test_y)
#plotBestFit(train_x,train_y,w)