import numpy as np
import random
import networkx as nx
import sys
from matplotlib import pyplot as plt
import scipy.stats as stats
import os
import ast  ## string list to list
import copy
import re
def getChildren(E, parent):
    '''
    如果E不严格，那个输出的children顺序（左右）也不严格，用sort也没用
    :param E:
    :param parent:
    :return:
    '''
    children = []
    for edge in E:
        if edge[0] == parent:
            children.append(edge[1])
    return children
def getChilrenByVTree(VTree,parent):
    '''
    20190308
    :param VTree:
    :param parent:
    :return:
    '''
    children = []
    for i in range(len(VTree)):
        if parent == VTree[i]:
            children.append(i+1)
    return children

def getParent(E, child):
    for edge in E:
        if edge[1] == child:
            return edge[0]
    return -1  # 没有父节点

def getDescendants(E, ancestor):
    descendants = []
    children = getChildren(E, ancestor)
    while children:
        parent = children[0]
        children1 = getChildren(E,parent)
        if children1:
            children.extend(children1)
        else:
            descendants.append(parent)
        del children[0]
    return descendants

def getAncestor(E, inode, dnode):
    child = inode
    parent = getParent(E, inode)
    while parent != -1:
        if isDescendant(E, dnode, parent):
            return parent
        else:
            parent = getParent(E, parent)
    # 无共同祖先节点
    return -1

def isDescendant(E, descendant, ancestor):
    child = descendant
    parent = getParent(E, child)
    while parent != -1:
        if ancestor == parent:
            return True
        else:
            parent = getParent(E, parent)
    return False

def Variance_way1(X):
    import numpy as np
    X = np.array(X)
    meanX = np.mean(X)
    n = np.shape(X)[0]
    # 按照协方差公式计算协方差，Note:分母一定是n-1
    variance = sum(np.multiply(X - meanX, X - meanX)) / (n-1)
    return variance

def GenTree(outDegree,pathNum):
    '''

    :param outDegree:
    :param pathNum:
    :return: VTree
    '''
    #linkSet=[0,1]
    linkSet = np.array([[0,1]])
    nodeNum=1
    #currentNodeSet=[1]
    currentNodeSet = np.array([[1]])
    flag=0
    pn=0;

    if outDegree > pathNum:
        print("出度要小于等于路径数\n")

    while pn < pathNum:
        #nextNodeSet = []
        nextNodeSet = np.array([[]])
        tempPN = pn
        addedNode = 0
        odegree = 0

        for i in range(len(currentNodeSet)):
            if currentNodeSet.size == 1:
                odegree = 2
            else:
                while True:
                    if pathNum - tempPN <3:
                        odegree = random.randint(1,3)
                    else:
                        odegree = random.randint(1,outDegree)
                    if tempPN+odegree+len(currentNodeSet)-i-1<=pathNum:
                        break;
            tempPN = tempPN+odegree

            if odegree > 1:
                flag = 1
                #childNodeSet =  range(nodeNum+1+addedNode,nodeNum+addedNode+odegree+1)
                childNodeSet = np.arange(nodeNum+1+addedNode,nodeNum+addedNode+odegree+1,1).reshape(1,odegree)
                if nextNodeSet.size == 0:
                    nextNodeSet = childNodeSet.T
                else:
                    nextNodeSet = np.r_[nextNodeSet,childNodeSet.T]
                tempNodeSet1 = np.tile(currentNodeSet[i,:],(odegree,1))
                tempNodeSet2=np.c_[tempNodeSet1,childNodeSet.T]
                linkSet = np.r_[linkSet, tempNodeSet2]
                #linkSet = np.r_[linkSet,np.c_[np.tile(currentNodeSet[i,:],(odegree,1)),childNodeSet.T]]
                addedNode = addedNode + odegree
            else:
                pn=pn+1
        if not flag:
            flag = 0;
            pn = pn - currentNodeSet.size;
            continue
        else:
            flag=0
            nodeNum = nodeNum +addedNode
            currentNodeSet=nextNodeSet;
        if pathNum-currentNodeSet.size == pn:
            pn=pathNum
    VTree=linkSet[:,0]
    return VTree



def getInternalNodes(VTree):
    internalNodes = []
    for i in VTree:
        internalNodes.append(i)
    return internalNodes

def getLeafNodes(VTree):
    '''

    :param VTree:
    :return: leafNodes
    '''
    leafNodes = []
    for i in range(len(VTree)):
        leafNode = i+1
        if leafNode not in VTree:
            leafNodes.append(leafNode)
    return leafNodes


def gen_linkDelay(VTree,scale = 1.0,probesNum = 200):
    '''
    产生链路时延
    :return:
    '''

    linkDelay = []
    for i in range(len(VTree)):
        linkDelay.append(np.random.exponential(scale, probesNum))
    return linkDelay

def calPathDelay(VTree, linkDelay):
    '''
    计算路径时延
    :return:
    '''
    PathDelay = []
    for i in range(len(VTree)):
        if i+1 not in VTree:
            tempsum = linkDelay[i]
            j = i+1
            while VTree[j-1] != 0:
                j = VTree[j-1]
                tempsum =tempsum+linkDelay[j-1]
            PathDelay.append(tempsum)
    return PathDelay

def Covariance_way2(X,Y):
    '''
    向量中心化方法计算两个等长向量的协方差convariance
    '''
    X, Y = np.array(X), np.array(Y)
    n = np.shape(X)[0]
    centrX = X - np.mean(X)
    centrY = Y - np.mean(Y)
    convariance = sum(np.multiply(centrX, centrY)) / (n - 1)
    # print('向量中心化方法求得协方差:', convariance)
    return convariance

def VTreetoE(VTree):
    E = []
    child = 1
    for parent in VTree:
        E.append((parent, child))
        child = child + 1
    return E

def numberTreeOld(E, root='s'):
    assert isinstance(E, list)
    newE = []  # 新的边list
    number = 0  # 正在处理的节点
    count = 1  # 已经出现的节点个数
    parents = [root]  # 待处理的节点
    while parents:
        parent = parents.pop()  # 将要处理的节点
        newParent = number  # 该节点的新的编号
        children = getChildren(E, parent)
        parent = number
        for child in children:
            parents.insert(0, child)
            newE.append((newParent, count))
            count = count+1
        number = number+1
        if count == len(E)+1:  # 当出现的节点个数符合要求就可以终止处理了
            break
    return newE





def getRM(D, VTree):
    '''
    产生路由矩阵
    :param D:[4,5,6,7,8]
    :param VTree:[0,1,1,2,2,2,3,3]
    :return:
    '''
    pathNum = len(D)
    linkNum = len(VTree)
    A = np.zeros((pathNum,linkNum))
    for i in range(pathNum):
        leafNode = D[i]
        A[i][leafNode-1] = 1
        parentNode = VTree[leafNode-1]
        while parentNode != 0:
            A[i][parentNode-1] = 1
            parentNode = VTree[parentNode-1]
    return A

def numberTopo(E,R):
    '''
    根据路径编号
    :param E:
    :param R:
    :return:newE
    '''
    transfer = {} ##内部节点转换表
    newE = []  ##具有一定顺序的边集合
    number = len(R)+1  ##编号
    for leafnode in R:
        trace = []  #一条路径的轨迹
        parent = getParent(E,leafnode)
        trace.append(parent)
        cnt = 0
        while getParent(E,parent) != 0:
            cnt += 1
            if cnt > 100:
                print("死循环",E)
                sys.exit(0)
            parent = getParent(E,parent)
            trace.append(parent)
        trace.reverse()
        for i in range(len(trace)):
            if trace[i] not in transfer:
                transfer[trace[i]] = number
                number = number+1
                trace[i] = transfer[trace[i]]
            else:
                trace[i] = transfer[trace[i]]
        tempE = []
        if (0,trace[0]) not in newE:
            tempE.append((0,trace[0]))
        if len(trace ) == 1:
            if (trace[0],leafnode) not in newE:
                tempE.append((trace[0],leafnode))

        else:
            index = 0
            while index < len(trace)-1:
                if (trace[index], trace[index+1]) not in newE:
                    tempE.append((trace[index], trace[index+1]))
                index = index+1
            if (trace[len(trace)-1],leafnode) not in newE:
                tempE.append((trace[len(trace)-1],leafnode))
        tempE.reverse()
        newE.extend(tempE)
    return newE



def EtoVTree(E):
    i = 1
    VTree = []
    while getParent(E,i) != -1:
        parent = getParent(E,i)
        VTree.append(parent)
        i = i+1
    return VTree


def to_zzzNode(E,root=0):
    from zss import Node
    A = Node(str(root))
    U = [(0,A)]
    while len(U) != 0:
        parent = U[0][0]
        node = U[0][1]
        del U[0]
        children = getChildren(E, parent)
        if len(children) == 0:
            continue
        for i in range(len(children)):
            node.addkid(Node(str(children[i])))
            U.append((children[i], node.children[i]))
    return A

def calEDbyzss(E1,E2,root=0):
    '''
    计算编辑距离
    :param E1:
    :param E2:
    :param root:
    :return:
    '''
    from zss import simple_distance
    A1 = to_zzzNode(E1)
    A2 = to_zzzNode(E2)
    return simple_distance(A1,A2)



def VTree2ToVTree1(VTree2,root=0):
    '''
    第二版VTree改为第一版，转换可能导致拓扑结构不一样
    :param VTree2:  [6,6,7,7,0,5,5]
    :return: [0,1,1,2,2,3,3]
    '''
    number = 0
    changeTable = {} ##对换表
    newE = [] ##不严谨的E set
    visit=[root]
    while visit:
        node = visit[0]
        del visit[0]
        if node not in changeTable:
            changeTable[node] = number
            number +=1
        children = getChilrenByVTree(VTree2,node)
        visit.extend(children)
        for child in children:
            newE.append((changeTable[node],number))
            changeTable[child] = number
            number += 1
    VTree1 = EtoVTree(newE)
    return VTree1






if __name__ == "__main__":
    E1 = {(0,6),(6,7),(6,8),(7,1),(7,2),(8,3),(8,4),(8,5)}
    E2 = {(0, 6), (6, 7), (6, 8), (7, 1), (7, 2), (8, 9),(9,3), (9, 4), (8, 5)}
    dis = calEDbyzss(E1,E2)
    print(dis)