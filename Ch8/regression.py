import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import json
import urllib2
from bs4 import BeautifulSoup

# train format is mat
def loadDataSet(filename):
    fr = open(filename)
    lines = fr.readlines()
    len_row = len(lines)
    len_col = len(lines[0].strip().split())
    train_x = np.mat(np.zeros((len_row,len_col-1)))
    train_y = []
    index_row = 0
    for loop_line in lines:
        loop_line = loop_line.strip().split()
        #train_x[index_row,0] = 1.0
        train_x[index_row,:] = loop_line[:-1]
        train_y.append(float(loop_line[-1]))
        index_row += 1
    train_y = np.mat(train_y).T
    #print(train_x,train_y)
    return train_x,train_y

#pending
def lgLoadData(set_num, year, num_piece, ori_pri):
    train_x = []
    train_y = []
    #sleep(5)
    #my_api_str = 'get from code.google.com'
    #data_url = 'https://www.googleapis.com/shopping/search/v1/public/product?\
    #key=%s&country=US&q=lego+%d&alt=json' % (my_api_str, set_num)
    #url = 'file:///Users/mhl/Documents/ML/machinelearninginaction/Ch08/setHtml/lego8288.html'
    #pg = urllib2.urlopen(url)
    #print pg.read()
    inFile = '/Users/mhl/Documents/MhlCode/mla/Ch8/setHtml/lego10196.html'
    fr = open(inFile, 'r')
    soup = BeautifulSoup(fr.read(), "html.parser")
    print (soup)
    i = 1
    current_row = soup.find_all('table', r='%d' % i)
    while (len(current_row) != 0):
        #print (current_row[0].findAll('a'))
        title = current_row[0].findAll('a')[1].text
        print (title)
        title_low = title.lower()
        print (title_low.find('new'))
        i += 1
        current_row = soup.find_all('table', r='%d' % i)
        print (current_row)
    #data_dict = json.loads(pg.read())

    return

# solve w
def getWeights(train_x, train_y):
    xTx = train_x.T*train_x
    if (np.linalg.det(xTx) == 0.0):
        print 'wrong, the det is zero, can not calculate I'
        return 1
    w = xTx.I*train_x.T*train_y
    #print (w)
    return w

# draw the data
def fig(train_x, train_y, w):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # change the col vector to row one
    ax.scatter(train_x[:, 1].flatten().A[0], train_y.flatten().A[0])
    ax.plot(train_x[:, 1].flatten().A[0], (train_x*w).flatten().A[0], c='r')
    plt.show()

def fig2(train_x, train_y, p_y):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # change the col vector to row one
    ax.scatter(train_x[:, 1].flatten().A[0], train_y.flatten().A[0])
    # let train_x sort
    sort_index = train_x[:, 1].argsort(0)
    temp_x = train_x[sort_index, 1]
    print(sort_index, temp_x)
    ax.plot(temp_x.flatten().A[0], p_y[sort_index].flatten().A[0], c='r')
    plt.show()

def fig3(w_lam, num_iter):
    x = range(num_iter)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(8):
        ax.plot(x, w_lam[i, :].A[0])
    plt.show()

# local weight line regression
def lwlrGetW(train_x, test_item, k):
    m, n= train_x.shape
    w = np.mat(np.eye(m))
    for i in range(m):
        diff = train_x[i] - test_item
        w[i, i] = np.exp(diff*diff.T/(-2*(k**2)))
    #print w
    xTWx = train_x.T*w*train_x
    if (np.linalg.det(xTWx) == 0):
        print 'wrong, det is zero'
        return
    return w, xTWx

def lwlrItem(train_x, train_y, test_item, k):
    w, xTWx = lwlrGetW(train_x, test_item, k)
    w_hat = xTWx.I*train_x.T*w*train_y
    #print (w_hat)
    return w_hat

def lwlrArray(train_x, train_y, test_x, k=1):
    m, n = test_x.shape
    w_all = np.mat(np.zeros((n, m)))
    p_y = np.mat(np.zeros((m, 1)))
    for i in range(m):
        w_all[:, i] = lwlrItem(train_x, train_y, test_x[i], k)
        p_y[i] = test_x[i]*w_all[:, i]
    #print (w_all, p_y)
    return p_y

def rssError(p_y, test_y):
    diff = p_y - test_y
    return (diff.T*diff)

def ridgeGetW(train_x, train_y, lam=0.2):
    xTxLamI = train_x.T*train_x+lam*np.eye(train_x.shape[1])
    if (np.linalg.det(xTxLamI) == 0.0):
        print 'Wrong, det is zero'
        return
    w = xTxLamI.I*train_x.T*train_y
    return w

def normData(input_x, input_y):
    x_mean = np.mean(input_x, 0)
    x_var = np.var(input_x, 0)
    x_norm = (input_x-x_mean)/np.sqrt(x_var)
    y_mean = np.mean(input_y, 0)
    y_norm = input_y - y_mean
    return x_norm, y_norm

def ridgeLamTry(train_x, train_y, numIter=30):
    train_x, train_y = normData(train_x, train_y)
    m, n = train_x.shape
    w_lam = np.mat(np.zeros((n, numIter)))
    for i in range(numIter):
        w_lam[:, i] = ridgeGetW(train_x, train_y, np.exp(i-10))
    print (w_lam)
    return w_lam

def stageWist(train_x, train_y, numIter=50, step_size=0.01):
    train_x, train_y = normData(train_x, train_y)
    m, n = train_x.shape
    w = np.zeros((n, 1))
    w_max = np.zeros((n, 1))
    lowest_error = 100000
    w_iter = np.mat(np.zeros((n, numIter)))
    for i in range(numIter):
        for j in range(n):
            for sign in [1, -1]:
                temp = w.copy()
                temp[j] = w[j]+step_size*sign
                p_y = train_x*temp
                rss_error = rssError(p_y, train_y)
                if (rss_error < lowest_error):
                    #w_max = temp
                    w[j] = temp[j]
                    lowest_error = rss_error
        #w = w_max
        w_iter[:, i] = np.mat(w)
    print (w_iter)
    return (w_iter)

# main
path = '/Users/mhl/Documents/MhlCode/mla/Ch8/'
filename = path+'abalone.txt'
input_x, input_y = loadDataSet(filename)
#w = getWeights(train_x, train_y)
#fig(train_x, train_y, w)
#test_item = [1, 2.0]
#lwlrGetW(train_x, test_item, 2)
'''
path = '/Users/mhl/Documents/MhlCode/mla/Ch8/'
filename = path+'ex0.txt'
train_x, train_y = loadDataSet(filename)
fig2(test_x, test_y, p_y)
'''
'''
p_y_01 = lwlrArray(input_x[0:99], input_y[0:99], input_x[0:99], 0.1)
p_y_1 = lwlrArray(input_x[0:99], input_y[0:99], input_x[0:99], 1)
p_y_10 = lwlrArray(input_x[0:99], input_y[0:99], input_x[0:99], 10)
e_y_01 = rssError(p_y_01, input_y[0:99])
e_y_1 = rssError(p_y_1, input_y[0:99])
e_y_10 = rssError(p_y_10, input_y[0:99])
print 'train set rssError'
print (e_y_01, e_y_1, e_y_10)
p_y_01 = lwlrArray(input_x[0:99], input_y[0:99], input_x[100:199], 0.1)
p_y_1 = lwlrArray(input_x[0:99], input_y[0:99], input_x[100:199], 1)
p_y_10 = lwlrArray(input_x[0:99], input_y[0:99], input_x[100:199], 10)
e_y_01 = rssError(p_y_01, input_y[100:199])
e_y_1 = rssError(p_y_1, input_y[100:199])
e_y_10 = rssError(p_y_10, input_y[100:199])
print 'test set rssError'
print (e_y_01, e_y_1, e_y_10)
'''
#w_lam = ridgeLamTry(input_x, input_y, 30)
#fig3(w_lam, 30)
#w_iter = stageWist(input_x, input_y, numIter=1000, step_size=0.01)
#fig3(w_iter, 1000)
lgLoadData(8288, 2006, 800, 49.99)