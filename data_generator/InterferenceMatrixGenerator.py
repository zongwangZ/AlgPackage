#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'InterferenceMatrixGenerator'.py
@ide     : 'PyCharm'
@time    : '2020'-'11'-'13' '18':'44':'56'
@contact : zongwang.zhang@outlook.com
'''
from network import Network
import numpy as np


class InterferenceMatrixGenerator:

    def __init__(self, topoloy: Network,p_correct):
        self._p_correct = p_correct
        self.__topoloy = topoloy
        self.__gen_matrix()

    def getInterferenceMatrix(self):
        return self.__interference_matrix

    def __gen_matrix(self):
        tunnel_list = []
        for node1 in self.__topoloy.getOverlayNodeSet():
            for node2 in self.__topoloy.getOverlayNodeSet():
                if node1 != node2:
                    tunnel_list.append([node1, node2])
        tunnel_num = len(tunnel_list)
        matrix = np.zeros(shape=(tunnel_num, tunnel_num), dtype=int)
        tunnel_path = []
        for tunnel in tunnel_list:
            tunnel_path.append(self.__topoloy.getShortestPath(tunnel[0],tunnel[1]))
        for tunnel1 in range(tunnel_num):
            for tunnel2 in range(tunnel_num):
                if tunnel1 < tunnel2:
                    path1 = tunnel_path[tunnel1]
                    path2 = tunnel_path[tunnel2]
                    matrix[tunnel1][tunnel2] = matrix[tunnel2][tunnel1] = self.gen_interference_with_p(path1,path2)

                elif tunnel1 == tunnel2:
                    matrix[tunnel1][tunnel2] = 0
        self.__interference_matrix = matrix

    def gen_interference_with_p(self,path1,path2):
        """
        暂时
        根据概率来判断是否得到正确的结果
        :param path1:
        :param path2:
        :return:
        """
        result = self.__topoloy.isIntersect(path1, path2)
        fake_result = 1 if self.__topoloy.isIntersect(path1,path2) == 0 else 0
        return result if np.random.random() < self._p_correct else fake_result
