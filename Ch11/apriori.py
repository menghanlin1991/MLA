import numpy as np
import matplotlib.pyplot as plt
import time

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(data_set):
    C1 = []
    for tran in data_set:
        for item in tran:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    #print (C1)
    return map(frozenset, C1)

def getFreSupportVal(data_set, Ck, min_support):
    Ck_support_val = {}
    num_tran = len(data_set)
    for chk in Ck:
        for tran in data_set:
            if chk.issubset(tran):
                if chk not in Ck_support_val.keys():
                    Ck_support_val[chk] = 0
                Ck_support_val[chk] += 1
    ret_list = []
    for key in Ck_support_val.keys():
        temp = Ck_support_val[key] / float(num_tran)
        Ck_support_val[key] = temp
        if temp >= min_support:
            ret_list.append(key)
    #print (ret_list, Ck_support_val)
    return ret_list, Ck_support_val

#k >= 1, L is frozenset
def aprioriGenC(L, k):
    ret_list = []
    for i in range(len(L)):
        for j in range(i+1, len(L)):
            L1 = list(L[i])[:k-1]; L2 = list(L[j])[:k-1]
            #one thing is each item, no same L will happen
            if (k == 1):
                ret_list.append(L[i] | L[j])
            else:
                if (L1[0:k-1] == L2[0:k-1]):
                    ret_list.append(L[i] | L[j])
    #print (ret_list)
    return ret_list

def apriori(data_set, min_support):
    C1 = createC1(data_set)
    #record the frequent set
    fre_record = []
    fre_val_merge = {}
    L1, C1_support_val = getFreSupportVal(data_set, C1, min_support)
    fre_record.append(L1)
    fre_val_merge = dict(C1_support_val.items() + fre_val_merge.items())
    Lk = L1
    k = 1
    while (1):
        Ck_plus_1 = aprioriGenC(Lk, k)
        Lk_plus_1, Ck_plus_1_support_val = getFreSupportVal(data_set, Ck_plus_1, min_support)
        if (len(Lk_plus_1) == 0):
            break
        fre_record.append(Lk_plus_1)
        fre_val_merge = dict(Ck_plus_1_support_val.items() + fre_val_merge.items())
        Lk = Lk_plus_1
        k += 1
    print (fre_record, fre_val_merge)
    return fre_record, fre_val_merge

def getRuleLk(L, fre_val_merge, min_conf=0.7):
    rule = []
    for item in L:
        ret_rule = getRuleOfOneItem(item, fre_val_merge, min_conf)
        if (len(ret_rule)!=0):
            rule.append(ret_rule)
    return rule

def getRuleOfOneItem(item, fre_val_merge, min_conf):
    #print (item)
    ret_rule = []
    m = len(item)
    H1 = [frozenset([i]) for i in item]
    k = 1
    H = H1
    while (k < m):
        for i in H:
            set_r = i
            set_l = item - i
            conf = fre_val_merge[item]/fre_val_merge[set_l]
            if (conf >= min_conf):
                ret_rule.append((set_l, set_r, conf))
            else:
                H = list(frozenset(H) - set_r)
        if (len(H) == 0):
            break
        H = aprioriGenC(H, k)
        k += 1
    #print ('ret_rule for item', item, ret_rule)
    return ret_rule

def getRule (fre_record, fre_val_merge, min_conf=0.7):
    ret_rule = []
    for Lk in fre_record:
        #print ('lk[0]', Lk[0])
        if (len(Lk[0]) == 1):
            continue
        ret_rule.append(getRuleLk(Lk, fre_val_merge, min_conf))
    print ('rule', ret_rule)
    return ret_rule

def mushroomTest(filename):
    mush_dat_set = [line.split() for line in open(filename).readlines()]
    #print (mush_dat_set)
    fre_record, fre_val_merge = apriori(mush_dat_set, 0.5)
    print (fre_record)
    '''
    print (time.time())
    getRule(fre_record, fre_val_merge, min_conf=0.7)
    print (time.time())
    '''
    return
'''
data_set = loadDataSet()
C1 = createC1(data_set)
L1, C1_support_val = getFreSupportVal(data_set, C1, 0.3)
k = len(L1[0])
aprioriGenC(L1, k)

data_set = loadDataSet()
fre_record, fre_val_merge = apriori(data_set, 0.5)
print (time.time())
getRule(fre_record, fre_val_merge, min_conf=0.7)
print (time.time())
'''

path = '/Users/mhl/Documents/MhlCode/mla/Ch11/'
filename = path+'mushroom.dat'
mushroomTest(filename)

