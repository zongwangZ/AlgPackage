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
import networkx as nx
from matplotlib import pyplot as plt
class Network:
    def __init__(self, overlay_node_num, edge_set):
        self.__overlay_node_num = overlay_node_num
        self.__edge_set = edge_set
        self.G = self.create_graph()

    def create_graph(self):
        G = nx.Graph()
        for edge in self.__edge_set:
            G.add_edge(edge[0],edge[1])
        # nx.draw(G, with_labels=True)
        # plt.show()
        return G

    def getEdgeSet(self):
        """
        返回边集合，无向
        :return:
        """
        return self.__edge_set

    def getOverlayNodeNum(self):
        """
        返回边缘节点的数量
        :return:
        """
        return self.__overlay_node_num

    def getAdjacentNodes(self,node):
        """
        获取与node邻接的所有节点
        :param node:
        :return:
        """
        adjacent_nodes = []
        for edge in self.__edge_set:
            if edge[0] == node:
                adjacent_nodes.append(edge[1])
            if edge[1] == node:
                adjacent_nodes.append(edge[0])
        return adjacent_nodes

    def getShortestPath(self, node1, node2):
        """
        最短路由
        :return:
        """
        path = []
        node_set = nx.shortest_path(self.G, node1,node2)
        for i in range(len(node_set)-1):
            path.append((node_set[i],node_set[i+1]))
        return path



    def isIntersect(self, path1, path2):
        """
        判断两个路径是否相交
        :param path1:
        :param path2:
        :return:
        """
        flag = 0
        for edge in path1:
            if edge in path2:
                flag = 1
                break
        return flag