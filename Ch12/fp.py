import numpy as np
import matplotlib.pyplot as plt

class treeNode:
    def __init__(self, name, val, parent):
        self.name = name
        self.val = val
        self.parent = parent
        self.children = {}
        self.nodelink = None

    def inc(self):
        self.val += 1

    def disp(self, deep=0):
        print ' '*deep*4, self.name, ':', self.val
        if (len(self.children) == 0):
            return
        deep += 1
        for child in self.children.values():
            child.disp(deep)

def genFPTree(data_set, min_sup=3):
    #print (data_set)
    header_table = {}
    C1, L1 = getFreSupVal(data_set, min_sup)
    for key in C1:
        header_table[key] = [C1[key], None]
    ret_list, count_list = delSortL1(data_set, L1)
    if (len(ret_list) == 0):
        root_node = None
        return header_table, root_node
    root_node = treeNode('Null Set', 1, None)
    index = 0
    for tran in ret_list:
        count = count_list[index]
        addTranToTree(tran, count, root_node, header_table)
        index += 1
    #print ('header_table', header_table)
    return header_table, root_node

def addTranToTree(tran, count, root_node, header_table):
    #print (tran)
    item = tran[0]
    #print ('tran', tran)
    if item in root_node.children.keys():
        root_node.children[item].inc()
    else:
        root_node.children[item] = treeNode(item, count, root_node)
        if (header_table[item][1] == None):
            header_table[item][1] = root_node.children[item]
        else:
            updateNodeLink(header_table, item, root_node.children[item])
    if (len(tran) > 1):
        root_node = root_node.children[item]
        addTranToTree(tran[1:], count, root_node, header_table)
    return

def updateNodeLink(header_table, item, target_node):
    node = header_table[item][1]
    while (node.nodelink != None):
        node = node.nodelink
    node.nodelink = target_node

#first traversal
def getFreSupVal(data_set, min_sup):
    #print ('data_set', data_set)
    m = len(data_set)
    C1 = {}
    for tran in data_set.keys():
        for item in tran:
            if item not in C1.keys():
                C1[item] = 0
            C1[item] += data_set[tran]
    L1 = {}
    for key in C1.keys():
        #sup = C1[key] / float(m)
        sup = C1[key]
        if sup >= min_sup:
            L1[key] = sup
        else:
            del(C1[key])
    #print ('C1', C1)
    #print ('L1', L1)
    return C1, L1

#second traversal, del and sort
def delSortL1(data_set, L1):
    sort_list = []
    count_list = []
    for tran, count in data_set.items():
        temp = {}
        for item in tran:
            if item in L1.keys():
                temp[item] = L1[item]
        ret_order = [ret[0] for ret in sorted(temp.items(), key=lambda p:p[1], reverse=True)]
        sort_list.append(ret_order)
        count_list.append(count)
    #sort_list = map(frozenset, sort_list)
    #print ('ret_list', sort_list)
    return sort_list, count_list

#find the leafnode path
def ascendTree(leaf_node):
    path = []
    node = leaf_node.parent
    while (node.parent != None):
        #print ('node.name', node.name)
        path.append(node.name)
        node = node.parent
    return path

def findPrefixPath(header_table, base_pat):
    if (len(header_table) < 1):
        return {}
    leaf_node = header_table[base_pat][1]
    path_dict = {}
    while (leaf_node != None):
        ret_path = ascendTree(leaf_node)
        if (len(ret_path) > 0):
            path_dict[frozenset(ret_path)] = leaf_node.parent.val
        leaf_node = leaf_node.nodelink
    #print (path_dict)
    return path_dict

def findFreSet(header_table, root_node, min_sup=3, pre_fix=set([]), freq_list=[]):
    header_list = [loop[0] for loop in sorted(header_table.items(), key=lambda p:p[1])]
    #print ('header_list', header_list)
    print ('freq_list', freq_list)
    for base_pat in header_list:
        new_fre_set = pre_fix.copy()
        new_fre_set.add(base_pat)
        freq_list.append(new_fre_set)
        pre_fix_path = findPrefixPath(header_table, base_pat)
        #print ('pre_fix_path', pre_fix_path)
        header_table_next, root_node_next = genFPTree(pre_fix_path, min_sup)
        if (root_node_next != None):
            findFreSet(header_table_next, root_node_next, min_sup, new_fre_set, freq_list)
    return freq_list

def loadSimpDat():
    simp_dat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simp_dat

def createDictSet(data_set):
    ret_dict = {}
    for tran in data_set:
        ret_dict[frozenset(tran)] = 1
    #print ('init_dict', ret_dict)
    return ret_dict
'''
root_node = treeNode('pyramid', 9, None)
root_node.children['eye'] = treeNode('eye', 13, root_node)
l_node = root_node.children['eye']
l_node.children['test'] = treeNode('test', 0, root_node)
root_node.children['phoenix'] = treeNode('phoenix', 3, root_node)
root_node.disp()

data_set = loadSimpDat()
C1, L1 = getFreSupVal(data_set, 0.5)
delSortL1(data_set, L1)
'''
data_set = loadSimpDat()
ret_dict = createDictSet(data_set)
header_table, root_node = genFPTree(ret_dict)
#root_node.disp(deep=0)
findPrefixPath(header_table, 'z')
freq_list = findFreSet(header_table, root_node)
#
