import numpy as np
import matplotlib.pyplot as plt

# train format is mat, input: none, output: train_x, train_y
def loadData():
    train_x = np.mat([[1., 2.1], [2., 1.1],
                      [1.3, 1.], [1., 1.],
                      [2., 1.]])
    train_y = np.mat([1.0, 1.0, -1.0, -1.0, 1.0]).T
    print(train_x, train_y)
    return train_x, train_y

# train format is mat
def loadDataSet(filename):
    fr = open(filename)
    lines = fr.readlines()
    len_row = len(lines)
    len_col = len(lines[0].strip().split())
    train_x = np.mat(np.zeros((len_row,len_col)))
    train_y = []
    index_row = 0
    for loop_line in lines:
        loop_line = loop_line.strip().split()
        train_x[index_row,0] = 1.0
        train_x[index_row,1:] = loop_line[:-1]
        train_y.append(int(float(loop_line[-1])))
        index_row += 1
    train_y = np.mat(train_y).T
    #print(train_x,train_y)
    return train_x,train_y

# get the estimate result of input: dim, threshold, cpr
def getStumpEst(train_x, dim, threshold, cpr):
    m, n = train_x.shape
    res_est = np.ones((m, 1))
    for index in range(m):
        if (cpr == 'lt'):
            if (train_x[index, dim] <= threshold):
                res_est[index] = -1
        elif (cpr == 'gt'):
            if (train_x[index, dim] > threshold):
                res_est[index] = -1
    #print ('dim:%d, threshold:%f, cpr:%s' % (dim, threshold, cpr))
    #print (res_est)
    return res_est

# get the best dim, threshold and cmp. input: train_x, train_y output: dim, threshold, cmp, res_est
def buildStump(train_x, train_y, D):
    m, n = train_x.shape
    best_stump = {}
    # i is dimension, j is threshold, k is compare
    error_min = 10000
    for i in range(n):
        min_dim = float(train_x[:, i].min())
        max_dim = float(train_x[:, i].max())
        num_step = 10
        step_size = float(max_dim-min_dim)/num_step
        #print (step_size)
        thre_pre = min_dim
        for j in range(num_step+1):
            for k in ['lt','gt']:
                res_est = getStumpEst(train_x, i, thre_pre, k)
                error_temp = 0
                for loop in range(m):
                    if (res_est[loop] != train_y[loop]):
                        error_temp += D[loop]
                #print (error_temp, res_est, error_min)
                if (error_temp < error_min):
                    error_min = error_temp
                    best_stump['dim'] = i
                    best_stump['thre'] = thre_pre
                    best_stump['cmp'] = k
                    best_res_est = res_est
            thre_pre = thre_pre + step_size
            #print(thre_pre)
    #print(best_stump, best_res_est, error_min)
    #print (best_res_est, error_min)
    return best_stump, best_res_est, error_min

# D is row vector, agg_Gx, best_res_est are col vector.
def adaBoost(train_x, train_y, num_iter):
    m, n = train_x.shape
    D = np.ones(m)/m
    Gx = np.zeros((train_x.shape[0], 1))
    best_stump_clu = []
    for i in range(num_iter):
        best_stump, best_res_est, error_min = buildStump(train_x, train_y, D)
        # get the Gx
        alpha_i = 0.5 * (np.log((1-error_min)/max(error_min, 1e-16)))
        Gx += alpha_i * best_res_est
        best_stump['alpha'] = alpha_i
        best_stump_clu.append(best_stump)
        # check if this is already ok
        error_list = np.multiply(np.sign(Gx) != train_y.A, np.ones((m,1)) )
        error_rate = error_list.sum()/m
        if ( error_rate == 0.0):
            break
        # update D
        temp_y = train_y.A
        temp = np.exp(-alpha_i * temp_y * best_res_est)
        z_m = (np.mat(D) * np.mat(temp)).A
        D = (np.multiply(D, temp.T) / z_m)[0]
        #print(best_res_est, error_min, alpha_i, temp_y, temp, z_m, D, Gx)
        #print (D)
    print (best_stump_clu)
    return best_stump_clu
# test, ret_test_y is col vector
def testAdaBoost(test_x, best_stump_clu):
    m, n = test_x.shape
    Gx = np.zeros((m, 1))
    num_fx = len(best_stump_clu)
    for i in range(num_fx):
        dim = best_stump_clu[i]['dim']
        thre = best_stump_clu[i]['thre']
        cmp = best_stump_clu[i]['cmp']
        f_x = getStumpEst(test_x, dim, thre, cmp)
        Gx += best_stump_clu[i]['alpha'] * f_x
    ret_test_y = np.sign(Gx)
    #print (ret_test_y)
    return ret_test_y

def errorRate(ret_test_y, test_y):
    temp = [ret_test_y != test_y]
    print (temp)
    temp = np.array(temp)
    error_rate = float(temp.sum())/test_y.shape[0]
    return error_rate


# main
'''
train_x, train_y = loadData()
#D = np.ones((train_x.shape[0]))/train_x.shape[0]
#print(D)
#buildStump(train_x, train_y, D)
#best_stump_clu = adaBoost(train_x, train_y, 10)
test_x = np.mat([1.1, 1.2])
testAdaBoost(test_x, best_stump_clu)
'''
path = '/Users/mhl/Documents/MhlCode/mla/Ch7/'
filename = path+'horseColicTraining2.txt'
train_x,train_y = loadDataSet(filename)
filename = path+'horseColicTest2.txt'
test_x,test_y = loadDataSet(filename)
best_stump_clu = adaBoost(train_x, train_y, 10)
ret_test_y = testAdaBoost(test_x, best_stump_clu)
error_rate = errorRate(ret_test_y, test_y)
print(error_rate)

