#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'RNJAlg'.py
@ide     : 'PyCharm'
@time    : '2021'-'03'-'13' '20':'13':'29'
@contact : zongwang.zhang@outlook.com
'''
"""
新版RNJ算法
"""
from network import Network
import numpy as np
from util.tool import numberTopo
from util.tool import EtoVTree
from util.tool import VTreetoE
class RNJAlg:
    def __init__(self,network:Network,logger=None):
        self.topo = network
        self.init_logger(logger)

    def inference(self,similarity_metrics,threshold=0.5):
        S = similarity_metrics
        e = threshold
        R = self.topo.getDestinations()
        dotR = R.copy()
        V = [0]
        E = []
        n = len(dotR)
        number = n + 1
        pathDistance = []
        for i in range(n):
            pathDistance.append(S[i][i])
            S[i][i] = 0

        while len(dotR) != 1:
            for node in R:
                if node in R and node not in dotR:
                    for i in range(len(S[0])):
                        S[R.index(node)][i] = S[i][R.index(node)] = 0

            indexs = np.where(np.max(S) == S)
            iIndex = indexs[0][0]
            jIndex = indexs[1][0]
            R.append(number)
            dotR.remove(R[iIndex])
            dotR.remove(R[jIndex])
            V.append(R[iIndex])
            V.append(R[jIndex])
            E.append((number, R[iIndex]))
            E.append((number, R[jIndex]))
            brother = []
            for kNode in dotR:
                kIndex = R.index(kNode)
                if S[iIndex][jIndex] - S[iIndex][kIndex] < e:
                    brother.append(kNode)
                    V.append(kNode)
                    E.append((number, kNode))
            for node in brother:
                dotR.remove(node)
            n = len(R)
            tempS = np.zeros((n, n))
            for i in range(n - 1):
                for j in range(n - 1):
                    tempS[i][j] = S[i][j]
            S = tempS
            for node in dotR:
                index = R.index(node)
                S[n - 1][index] = S[index][n - 1] = S[iIndex][index]
            dotR.append(number)
            number = number + 1
        E.append((0, dotR[0]))
        inferredE = numberTopo(E, self.topo.getDestinations())
        inferred_vtree = EtoVTree(inferredE)
        inferredE = VTreetoE(inferred_vtree)
        return inferredE

    def init_logger(self,logger=None):
        if not logger:
            self.logger = logger
