import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Circle
from os import listdir

#Prepare Data
def loadDataSet(filename):
    fr = open(filename)
    lines = fr.readlines()
    num_row = len(lines)
    num_col = len(lines[0].strip().split())
    train_x = np.zeros((num_row,num_col-1))
    train_y = np.zeros((num_row,1))
    for i in range(num_row):
        temp_list = lines[i].strip().split()
        train_x[i,:] = temp_list[:-1]
        train_y[i] = temp_list[-1]
    fr.close()
    #print(train_x.shape,train_y.shape)
    return train_x,train_y

#Prapare Digits Data
def sizeImg(filename):
    fr = open(filename)
    lines = fr.readlines()
    num_row = len(lines)
    num_col = len(lines[0].strip())
    return num_row,num_col

def image2vector(filename,num_col):
    img_vec = []
    fr = open(filename)
    lines = fr.readlines()
    for loop_line in lines:
        for i in range(num_col):
            img_vec.append(loop_line[i])
    return img_vec

def prepareData(path):
    data_list = listdir(path)
    del(data_list[0])
    num_data = len(data_list)
    filename0 = path + data_list[0]
    num_row,num_col = sizeImg(filename0)
    data_x = np.zeros((num_data,num_row*num_col))
    data_y = []
    for i in range(num_data):
        filename = path+data_list[i]
        data_x[i] = image2vector(filename,num_col)
        temp = int(data_list[i].split('_')[0])
        temp = -1 if (temp==9) else 1
        data_y.append(temp)
    data_y = np.array(data_y)
    data_y = data_y.reshape((len(data_y),1))
    #print(data_x,data_y)
    return data_x,data_y

#Train
def randomChooseJ(i,m):
    j = i
    while (j == i):
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(alpha,L,H):
    if (alpha > H):
        alpha = H
    if (alpha < L):
        alpha = L
    return alpha

def smoSimple(train_x,train_y,C,toler,num_iter):
    #transfer to matrix
    train_x = np.mat(train_x)
    train_y = np.mat(train_y)
    print(train_x,train_y)
    m,n = train_x.shape
    #initail alpha and b, iter
    b = 0.0
    alpha = np.mat(np.zeros((m,1)))
    iter = 0
    while (iter <= num_iter):
        alpha_change_record = 0
        #choose alpha1 one by one
        for i in range(m):
            yi = float((np.multiply(alpha,train_y).T)*(train_x*train_x[i].T))+b
            Ei = yi - float(train_y[i])
            if ((train_y[i]*Ei>toler and alpha[i]>0) or (train_y[i]*Ei<toler and alpha[i]<C)):
                j = randomChooseJ(i,m)
                yj = float((np.multiply(alpha,train_y).T)*(train_x*train_x[j].T))+b
                Ej = yj - float(train_y[j])
                #record the old alpha value
                alpha_i_old = alpha[i]
                alpha_j_old = alpha[j]
                if (train_y[i]==train_y[j]):
                    L = max(0,alpha_i_old+alpha_j_old-C)
                    H = min(alpha_i_old+alpha_j_old,C)
                else:
                    L = max(0,alpha_j_old-alpha_i_old)
                    H = min(C,C+alpha_j_old-alpha_i_old)
                #no value can choose in this line
                if (L==H):
                    continue
                eta = train_x[i]*train_x[i].T+train_x[j]*train_x[j].T-\
                    2*train_x[i]*train_x[j].T
                if (eta <= 0):
                    print('%d eta <= 0' % eta)
                alpha_j_new_unc = alpha_j_old + train_y[j]*(Ei-Ej)/eta
                alpha_j_new = clipAlpha(alpha_j_new_unc,L,H)
                if (abs(alpha_j_new-alpha_j_old)<0.00001):
                    #print('j not move enough, continue')
                    continue
                alpha_i_new = alpha_i_old + train_y[i]*train_y[j]*(alpha_j_old-alpha_j_new)
                #print(alpha_j_new,alpha_i_new)
                b_i_new = -Ei-train_y[i]*(train_x[i]*train_x[i].T)*(alpha_i_new-alpha_i_old)-\
                    train_y[j]*(train_x[j]*train_x[i].T)*(alpha_j_new-alpha_j_old)+b
                b_j_new = -Ej-train_y[i]*(train_x[i]*train_x[j].T)*(alpha_i_new-alpha_i_old)-\
                    train_y[j]*(train_x[j]*train_x[j].T)*(alpha_j_new-alpha_j_old)+b
                if (alpha_i_new>0 and alpha_i_new<C):
                    b = b_i_new
                elif (alpha_j_new>0 and alpha_j_new<C):
                    b = b_j_new
                else:
                    b = (b_i_new+b_j_new)/2
                #change alpha list
                alpha[i] = alpha_i_new
                alpha[j] = alpha_j_new
                alpha_change_record += 1
            #print('iter:%d i:%d alpha_change_record:%d' % (iter,i,alpha_change_record))
        if (alpha_change_record == 0):
            iter += 1
        else:
            iter = 0
    return b,alpha

#mistake, this should be same with the one written in book\
#because you should calculate kernel of test_x, revise
def kernelTrans(train_x,item_x,kTup):
    m,n = train_x.shape
    k = np.mat(np.zeros((m,1)))
    if (kTup[0]=='lin'):
        k = train_x*item_x.T
    elif (kTup[0]=='rbf'):
        for i in range(m):
            deltaij = train_x[i,:]-item_x
            #print(deltaij)
            norm_deltaij = deltaij*deltaij.T
            k[i] = np.exp(-norm_deltaij/(2*kTup[1]**2))
    else:
        print 'now, we don\'t support this kernel'
    #print(k)
    return k

class optStruct:
    def __init__(self,train_x,train_y,C,toler,kTup):
        self.train_x = train_x
        self.train_y = train_y
        self.m = train_x.shape[0]
        self.C = C
        self.toler = toler
        self.alpha = np.mat(np.zeros((self.m,1)))
        #ECache need one column to record if it is valid, so (m,2)
        self.ECache = np.mat(np.zeros((self.m,2)))
        self.b = 0
        #add kernel matrix
        self.k = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.k[:,i] = kernelTrans(self.train_x,self.train_x[i,:],kTup)

def calcEk(oS,k):
    yk = float((np.multiply(oS.alpha, oS.train_y).T) * (oS.k[:,k])) + oS.b
    Ek = yk - float(oS.train_y[k])
    return Ek

#diff from book
def selectJ(i,Ei,oS):
    maxK = -1 #save index
    maxDeltaEk = 0 #save the most diff value
    oS.ECache[i] = [1,Ei]
    #validK is a array to record the none zeros index
    validKList = np.nonzero(oS.ECache[:0].A)[0]
    #except i, there is valid k can calculate
    if (len(validKList)>1):
        for loop in validKList:
            if (loop != i):
                tempDeltaEk = abs(oS.ECache[loop,1] - Ei)
                if (tempDeltaEk>maxDeltaEk):
                    maxDeltaEk = tempDeltaEk
                    maxK = loop
        j = maxK
        Ej = oS.ECache[j,1]
    else:
        j = randomChooseJ(i,oS.m)
        Ej = calcEk(oS,j)
    return j,Ej

def updateECache(oS,k):
    Ek = calcEk(oS,k)
    oS.ECache[k] = [1,Ek]

#diff from book
def innerUpdate(i,oS):
    Ei = calcEk(oS,i)
    if ((oS.train_y[i] * Ei > oS.toler and oS.alpha[i] > 0) \
                or (oS.train_y[i] * Ei < oS.toler and oS.alpha[i] < oS.C)):
        j,Ej = selectJ(i,Ei,oS)
        #record the old alpha value
        alpha_i_old = oS.alpha[i]
        alpha_j_old = oS.alpha[j]
        if (oS.train_y[i] == oS.train_y[j]):
            L = max(0, alpha_i_old + alpha_j_old - oS.C)
            H = min(alpha_i_old + alpha_j_old, oS.C)
        else:
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(oS.C, oS.C + alpha_j_old - alpha_i_old)
        #no value can choose in this line
        if (L == H):
            return 0
        eta = oS.k[i,i] + oS.k[j,j] - \
              2 * oS.k[i,j]
        if (eta <= 0):
            print('%d eta <= 0' % eta)
        alpha_j_new_unc = alpha_j_old + oS.train_y[j] * (Ei - Ej) / eta
        alpha_j_new = clipAlpha(alpha_j_new_unc, L, H)
        if (abs(alpha_j_new - alpha_j_old) < 0.00001):
            # print('j not move enough, continue')
            return 0
        alpha_i_new = alpha_i_old + oS.train_y[i] * oS.train_y[j] * (alpha_j_old - alpha_j_new)
        #print(alpha_j_new,alpha_i_new)
        b_i_new = -Ei - oS.train_y[i] * (oS.k[i,i]) * (alpha_i_new - alpha_i_old) - \
                  oS.train_y[j] * (oS.k[j,i]) * (alpha_j_new - alpha_j_old) + oS.b
        b_j_new = -Ej - oS.train_y[i] * (oS.k[i,j]) * (alpha_i_new - alpha_i_old) - \
                  oS.train_y[j] * (oS.k[j,j]) * (alpha_j_new - alpha_j_old) + oS.b
        if (alpha_i_new > 0 and alpha_i_new < oS.C):
            oS.b = b_i_new
        elif (alpha_j_new > 0 and alpha_j_new < oS.C):
            oS.b = b_j_new
        else:
            oS.b = (b_i_new + b_j_new) / 2
        # change alpha list
        oS.alpha[i] = alpha_i_new
        oS.alpha[j] = alpha_j_new
        #update Ecache of this pair
        updateECache(oS,i)
        updateECache(oS,j)
        return 1
    else:
        return 0

def smoP(train_x,train_y,C,toler,max_iter,kTup=('lin',1)):
    #define oS to save the key data
    train_x = np.mat(train_x)
    train_y = np.mat(train_y)
    oS = optStruct(train_x,train_y,C,toler,kTup)
    print(oS.m)
    iter = 0
    enterSet = True
    alpha_pair_change_cnt = 0
    while((iter < max_iter) and (enterSet or alpha_pair_change_cnt>0)):
        alpha_pair_change_cnt = 0
        if enterSet:
            for i in range(oS.m):
                alpha_pair_change_cnt += innerUpdate(i,oS)
                print 'enterSet****iter %d, i %d, alpha_pair_change_cnt %d:' \
                      % (iter,i,alpha_pair_change_cnt)
            iter += 1
        else:
            #choose alpha bigger than 0, smaller then C
            alpha_special_list = np.nonzero((oS.alpha.A>0) * (oS.alpha.A<C))[0]
            #print(alpha_special_list)
            for k in alpha_special_list:
                alpha_pair_change_cnt += innerUpdate(k,oS)
                print 'partSet****iter %d, k %d, alpha_pair_change_cnt %d:' \
                       % (iter,k,alpha_pair_change_cnt)
            iter += 1
        #first search enterSet, then change the alpha between 0 and C till not change\
        # , then search the enterSet again
        if enterSet:
            enterSet = False
        elif (alpha_pair_change_cnt == 0):
            enterSet = True
    return oS.b,oS.alpha

#Anylysize Train Result
def plotSplitLine(train_x,train_y,b,alpha):
    #calculate the line
    w = np.multiply(alpha, train_y).T * train_x
    w = matrix2List(w)
    b = int(b)
    #calculate the circle point
    #print(w,b)
    alpha = matrix2List(alpha.T)
    print(alpha)
    #plot
    num_sample = len(train_y)
    x0 = []; y0 = []
    x1 = []; y1 = []
    for i in range(num_sample):
        if train_y[i] == -1:
            x0.append(train_x[i][0])
            y0.append(train_x[i][1])
        else:
            x1.append(train_x[i][0])
            y1.append(train_x[i][1])
    #print(x0)
    #print(y0)
    fig = plt.figure(1,facecolor='w')
    ax = fig.add_subplot(111)
    ax.scatter(x0,y0,c='r',marker='s')
    ax.scatter(x1,y1,c='g')
    x_line = np.arange(0,10,0.2)
    y_line = (-b-w[0]*x_line)/w[1]
    #print(x_line.shape,y_line.shape)
    #ax.plot(x_line,y_line)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Circle Support Vectors')
    #draw support vector
    m,n = train_x.shape
    num_sup_vec = 0
    for i in range(m):
        if alpha[i]>0:
            circle = Circle((train_x[i,0],train_x[i,1]),0.05,facecolor='none',\
                   edgecolor=(0,0.8,0.8),linewidth=3,alpha=0.5)
            ax.add_patch(circle)
            num_sup_vec += 1
    print('the support vector number is %d' % num_sup_vec)
    plt.show()

#tranfer matrix[[1,2,3]] to [1,2,3]
def matrix2List(input):
    input = input.A
    input = list(input[0])
    return input

#Test
def testRBF(kTup=('rbf', 0.5)):
    #train
    path = '/Users/mhl/Documents/MhlCode/mla/Ch6/'
    filename = path + 'testSetRBF.txt'
    train_x, train_y = loadDataSet(filename)
    #all data in oS is matrix
    b,alpha = smoP(train_x,train_y,0.6,0.0001,100,kTup)
    plotSplitLine(train_x, train_y, b, alpha)
    train_x = np.mat(train_x)
    train_y = np.mat(train_y)
    m,n = train_x.shape
    error_cnt = 0
    #only choose alpha > 0
    alpha_support_index = np.nonzero(alpha.A>0)[0]
    alpha_support_mat = alpha[alpha_support_index]
    x_support_mat = train_x[alpha_support_index]
    y_support_mat = train_y[alpha_support_index]
    print 'the support vector number is %d' % alpha_support_mat.shape[0]
    #print(alpha_support_index,alpha_support_mat.shape,x_support_mat.shape,y_support_mat.shape)
    for i in range(m):
        kernel_vec = kernelTrans(x_support_mat,train_x[i,:],kTup)
        #print(kernel_vec.shape)
        y_predict = float((np.multiply(alpha_support_mat,y_support_mat).T)*kernel_vec)+b
        if (np.sign(y_predict)!=np.sign(train_y[i])):
            error_cnt += 1
    print 'error percent of train data is :%f' % (float(error_cnt)/m)
    #plotSplitLine(train_x.A,train_y.A,b,alpha)
    #test test data
    filename = path + 'testSetRBF2.txt'
    test_x, test_y = loadDataSet(filename)
    test_x = np.mat(test_x)
    test_y = np.mat(test_y)
    m,n = test_x.shape
    error_cnt = 0
    for i in range(m):
        kernel_vec = kernelTrans(x_support_mat,test_x[i,:],kTup)
        #print(kernel_vec.shape)
        y_predict = float((np.multiply(alpha_support_mat,y_support_mat).T)*kernel_vec)+b
        if(np.sign(y_predict)!=np.sign(test_y[i])):
            error_cnt += 1
    print 'error percent of test data is :%f' % (float(error_cnt)/m)

def test():
    path = '/Users/mhl/Documents/MhlCode/mla/Ch6/'
    filename = path + 'testSet.txt'
    train_x, train_y = loadDataSet(filename)
    b, alpha = smoP(train_x, train_y, 200, 0.0001, 1000, kTup=('lin', 1.3))
    train_x = np.mat(train_x)
    train_y = np.mat(train_y)
    m,n = train_x.shape
    error_cnt = 0
    #only choose alpha > 0
    alpha_support_index = np.nonzero(alpha.A>0)[0]
    alpha_support_mat = alpha[alpha_support_index]
    x_support_mat = train_x[alpha_support_index]
    y_support_mat = train_y[alpha_support_index]
    #print(alpha_support_index,alpha_support_mat.shape,x_support_mat.shape,y_support_mat.shape)
    for i in range(m):
        kernel_vec = kernelTrans(x_support_mat,train_x[i,:],kTup=('rbf', 1.3))
        #print(kernel_vec.shape)
        y_predict = float((np.multiply(alpha_support_mat,y_support_mat).T)*kernel_vec)+b
        if (np.sign(y_predict)!=np.sign(train_y[i])):
            error_cnt += 1
    print 'error percent of train data is :%f' % (float(error_cnt)/m)

#Digits Test
def testDigits(kTup=('rbf', 0.8)):
    #train
    path = '/Users/mhl/Documents/MhlCode/mla/Ch6/digits/trainingDigits/'
    train_x,train_y = prepareData(path)
    #all data in oS is matrix
    b,alpha = smoP(train_x,train_y,0.6,0.0001,100,kTup)
    train_x = np.mat(train_x)
    train_y = np.mat(train_y)
    m,n = train_x.shape
    error_cnt = 0
    #only choose alpha > 0
    alpha_support_index = np.nonzero(alpha.A>0)[0]
    alpha_support_mat = alpha[alpha_support_index]
    x_support_mat = train_x[alpha_support_index]
    y_support_mat = train_y[alpha_support_index]
    print 'the support vector number is %d' % alpha_support_mat.shape[0]
    #print(alpha_support_index,alpha_support_mat.shape,x_support_mat.shape,y_support_mat.shape)
    for i in range(m):
        kernel_vec = kernelTrans(x_support_mat,train_x[i,:],kTup)
        #print(kernel_vec.shape)
        y_predict = float((np.multiply(alpha_support_mat,y_support_mat).T)*kernel_vec)+b
        if (np.sign(y_predict)!=np.sign(train_y[i])):
            error_cnt += 1
    print 'error percent of train data is :%f' % (float(error_cnt)/m)
    #plotSplitLine(train_x.A,train_y.A,b,alpha)
    #test test data
    path = '/Users/mhl/Documents/MhlCode/mla/Ch6/digits/testDigits/'
    test_x, test_y = prepareData(path)
    test_x = np.mat(test_x)
    test_y = np.mat(test_y)
    m,n = test_x.shape
    error_cnt = 0
    for i in range(m):
        kernel_vec = kernelTrans(x_support_mat,test_x[i,:],kTup)
        #print(kernel_vec.shape)
        y_predict = float((np.multiply(alpha_support_mat,y_support_mat).T)*kernel_vec)+b
        if(np.sign(y_predict)!=np.sign(test_y[i])):
            error_cnt += 1
    print 'error percent of test data is :%f' % (float(error_cnt)/m)

#Main
'''
path = '/Users/mhl/Documents/MhlCode/mla/Ch6/'
filename = path + 'testSet.txt'
train_x,train_y = loadDataSet(filename)
#time_begin = time.strftime("%Y-%m-%d %H:%M:%S")
#b, alpha = smoSimple(train_x,train_y,0.6,0.001,40)
#plotSplitLine(train_x,train_y,b,alpha)
#time_end = time.strftime("%Y-%m-%d %H:%M:%S")
time_begin = time.time()
b,alpha = smoP(train_x,train_y,0.6 ,0.0001,100,kTup=('lin',1.3))
time_end = time.time()
print(time_end-time_begin)
'''
#test()
testRBF(kTup=('rbf', 0.4))
#plotSplitLine(train_x,train_y,b,alpha)
#oS = optStruct(np.mat(train_x), np.mat(train_y), 0.6, 0.0001, kTup=('lin',1.3))
#path = '/Users/mhl/Documents/MhlCode/mla/Ch6/digits/trainingDigits/'
#prepareData(path)

#testDigits(kTup=('lin', 10))
