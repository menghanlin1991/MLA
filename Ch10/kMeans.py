import numpy as np
import matplotlib.pyplot as plt
import urllib
import json

def loadDataSet(filename):
    data_list = []
    fr = open(filename)
    for i in fr.readlines():
        temp = i.strip().split('\t')
        data_list.append(map(float, temp))
    data_mat = np.mat(data_list)
    #print (dataMat)
    return data_mat

def plot(data_mat):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.scatter(data_mat[:, 0].A.T[0], data_mat[:, 1].A.T[0])
    plt.show()

def plotClass(data_mat, class_record, k):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    color = ['r', 'b', 'y', 'g']
    for cent in range(k):
        temp = data_mat[np.nonzero(class_record[:, 0].A==cent)[0]]
        #print temp
        ax.scatter(temp[:, 0].A.T[0], temp[:, 1].A.T[0], c=color[cent])
    plt.show()

#mat is row vector
def distEuclid(vec1_mat, vec2_mat):
    diff = vec1_mat - vec2_mat
    ret = diff * diff.T
    return float(np.sqrt(ret))

def randKCent(data_mat, k):
    m, n = data_mat.shape
    k_cent = np.mat(np.zeros((k, n)))
    for i in range(n):
        k_cent[:, i] = min(data_mat[:, i]) + float((max(data_mat[:, i]) - min(data_mat[:, i]))) * np.mat(np.random.rand(k, 1))
    #print (k_cent)
    return k_cent

def kMeans(data_mat, k, distMeth=distEuclid):
    m, n = data_mat.shape
    class_record = np.mat(np.zeros((m, 2)))
    k_cent = randKCent(data_mat, k)
    class_changed = True
    first_flag = True
    while (class_changed):
        class_changed = False
        for i in range(m):
            min_dis = np.Inf
            min_class_index = -1
            for j in range(k):
                dist_euclid = distMeth(data_mat[i, :], k_cent[j, :])
                if (dist_euclid < min_dis):
                    #print dist_euclid, j
                    min_dis = dist_euclid
                    min_class_index = j
            if first_flag:
                class_record[i, 0] = min_class_index
                class_changed = True
            else:
                if (class_record[i, 0] != min_class_index):
                    class_changed = True
                    class_record[i, 0] = min_class_index
            class_record[i, 1] = min_dis
        first_flag = False
        #updata k_cent
        for cent in range(k):
            k_temp = data_mat[np.nonzero(class_record[:, 0].A==cent)[0]]
            k_cent[cent, :] = np.mean(k_temp, axis=0)
        #print (k_cent)
        #print (class_record)
    return k_cent, class_record

#wrong, wrong, wrong. my god
def biKMeans(data_mat, k, distMeth=distEuclid):
    #using cent_list to record all the centroids
    cent_list = []
    #first cent is the average of all the samples
    cent0 = np.mean(data_mat, axis=0).tolist()[0]
    cent_list.append(cent0)
    #for the first split
    m, n = data_mat.shape
    class_record = np.mat(np.zeros((m, 2)))
    #loop if number of class less than k
    while (len(cent_list) < k):
        #print (len(cent_list), cent_list)
        #print ('class_record', class_record)
        min_sse = np.Inf
        for i in range(len(cent_list)):
            data_mat_to_split = data_mat[np.nonzero(class_record[:, 0].A==i)[0]]
            two_cent, two_class_record = kMeans(data_mat_to_split, 2, distMeth)
            #print ('two_class_record', two_class_record)
            sse_split = float(sum(two_class_record[:, 1]))
            sse_no_split = float(sum(class_record[np.nonzero(class_record[:, 0].A!=i)[0], 1]))
            sse_total = sse_split + sse_no_split
            #print ('i, sse_split, sse_no_split, sse_total', i, sse_split, sse_no_split, sse_total)
            if (sse_total < min_sse):
                best_split_index = i
                best_two_cent = two_cent
                best_two_class_record = two_class_record
                min_sse = sse_total
        #update class_record inf
        temp_row_index = np.nonzero(class_record[:, 0].A==best_split_index)[0]
        zero_row_index = temp_row_index[np.nonzero(best_two_class_record[:, 0].A==0)[0]]
        one_row_index = temp_row_index[np.nonzero(best_two_class_record[:, 0].A==1)[0]]
        #print (temp_row_index, zero_row_index, one_row_index)
        class_record[temp_row_index, :] = best_two_class_record
        class_record[zero_row_index, 0] = best_split_index
        class_record[one_row_index, 0] = len(cent_list)
        #update cent information
        cent_list[best_split_index] = best_two_cent[0, :].tolist()[0]
        cent_list.append(best_two_cent[1, :].tolist()[0])
        #print (cent_list, class_record)
        #plotClass(data_mat, class_record, 4)

    return np.mat(cent_list), np.mat(class_record)

#know little about yahoo api, so just imatate
def geoGrab(address, city):
    apiStem = 'http://where.yahooapis.com/geocode?'
    params = {}
    params['flags'] = 'J'
    params['appid'] = 'B7NlOY32'
    params['location'] = '%s, %s' % (address, city)
    url_params = urllib.urlencode(params)
    print (url_params)
    yahooApi = apiStem + url_params
    print (yahooApi)
    c = urllib.urlopen(yahooApi)
    return json.loads(c.read())

path = '/Users/mhl/Documents/MhlCode/mla/Ch10/'
filename = path+'testSet2.txt'
data_mat = loadDataSet(filename)
#randKCent(data_mat, 3)
#k_cent, class_record = kMeans(data_mat, 4)
#plot(data_mat)
#plotClass(data_mat, class_record, 4)
#k_cent, class_record = biKMeans(data_mat, 3)
#plotClass(data_mat, class_record, 4)
geoGrab('1 VA Center', 'Augusta, ME')
