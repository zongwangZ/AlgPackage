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

    def __init__(self, topoloy:Network):
        self.__topoloy = topoloy
        self.__interference_matrix = self.__gen_matrix()


    def __gen_matrix(self):
        overlay_node_num = self.__topoloy.getOverlayNodeNum();
        tunnel_num = overlay_node_num*(overlay_node_num-1)
        matrix = np.zeros(shape=(tunnel_num,tunnel_num),dtype=int)
        matrix = [[0,1,0,0,0,1],
                  [1,0,0,1,0,1],
                  [0,0,0,1,1,0],
                  [0,1,1,0,0,0],
                  [0,0,1,0,0,1],
                  [1,0,0,0,1,0]]
        self.__interference_matrix = matrix

    def getInterferenceMatrix(self):
        return self.__interference_matrix