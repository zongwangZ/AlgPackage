#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'TST-python'
@author  : '张宗旺'
@file    : 'RNJ'.py
@ide     : 'PyCharm'
@time    : '2020'-'04'-'23' '21':'06':'58'
@contact : zongwang.zhang@outlook.com
'''

import numpy as np
from src.measure import Measurement
from src.tool import *
import copy
import json
from matplotlib import pyplot as plt


class RNJ:

    @staticmethod
    def RNJ(data):

        R = copy.copy(data['R'])

        S = data['S']

        e = data['e']

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

            # indexs = np.where(np.max(S) == S)[0]
            #
            # iIndex = indexs[0]
            #
            # jIndex = indexs[1]

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

        inferredE = numberTopo(E, data['R'])

        inferred_vtree = EtoVTree(inferredE)
        inferredE = VTreetoE(inferred_vtree)
        data['inferredE'] = inferredE

        return E

    @staticmethod
    def getThreshold(link_delay, R, way):
        link_delay_var = {}
        assert isinstance(link_delay, dict)
        for key in link_delay:
            if key not in link_delay_var:
                link_delay_var[key] = []
            delays = link_delay[key]
            for delay in delays:
                link_delay_var[key].append(np.var(delay))
        min = np.Inf
        if way == 0:  # 从多次测量中选取最小的link
            for key in link_delay_var:
                if key + 1 not in R:
                    for item in link_delay_var[key]:
                        if item < min:
                            min = item
        elif way == 1:  # 从多次测量中取均值
            for key in link_delay_var:
                if key + 1 not in R:
                    item = np.mean(link_delay_var[key])
                    if item < min:
                        min = item
        return min / 2

    @staticmethod
    def genData(tree_vector):
        # TreePlot(tree_vector)
        measurement = Measurement(tree_vector)
        pair_measurement = measurement.get_all_pair_measurement(1000)
        R = getLeafNodes(tree_vector)
        S = np.zeros(shape=(len(R), len(R)))
        index = 0
        for i in range(len(R)):
            for j in range(len(R)):
                if j > i:
                    S[i][j] = S[j][i] = pair_measurement[index]
                    index += 1
        link_delay = measurement.link_delay
        e = RNJ.getThreshold(link_delay, R, 0)
        data = {
            'VTree': tree_vector,
            'E': VTreetoE(tree_vector),
            'R': R,
            'S': S,
            'e': e

        }
        return data

    @staticmethod
    def doSim():
        PC = []  ## 精度
        ED = []  ## 编辑距离
        vtrees = [[16, 17, 17, 17, 19, 19, 19, 18, 20, 20, 15, 15, 15, 15, 0, 15, 16, 15, 18, 15]]
        # vtrees = [[5, 5, 6, 6, 0, 5]]
        for vtree in vtrees:
            cnt = 0
            edit_distance = []
            for i in range(1):
                data = RNJ.genData(vtree)
                RNJ.RNJ(data)
                E = data['E']
                inferredE = data['inferredE']
                ed = calEDbyzss(E, inferredE)
                edit_distance.append(ed)
                if E == inferredE:
                    cnt = cnt + 1

            ED.append(np.mean(edit_distance))
            print(np.mean(edit_distance))
            PC.append(cnt / 100)
            print(cnt / 100)


if __name__ == '__main__':
    RNJ.doSim()
