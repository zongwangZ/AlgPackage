#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'Network'.py
@ide     : 'PyCharm'
@time    : '2020'-'11'-'13' '18':'47':'24'
@contact : zongwang.zhang@outlook.com
'''

class Network:
    def __init__(self, overlay_node_num, edge_set):
        self.__overlay_node_num = overlay_node_num
        self.__edge_set = edge_set

    def getEdgeSet(self):
        return self.__edge_set

    def getOverlayNodeNum(self):
        return self.__overlay_node_num