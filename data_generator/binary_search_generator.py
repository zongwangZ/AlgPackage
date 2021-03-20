#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'binary-search-generator'.py
@ide     : 'PyCharm'
@time    : '2021'-'03'-'10' '18':'48':'31'
@contact : zongwang.zhang@outlook.com
'''
"""
用于给二分算法生成数据用的文件
"""
from data_generator.M3Generator import M3Generator
from data.topo_zoo.init import topo_zoo_set
from network import Network
import logging
import networkx as nx
import numpy as np
from util.tool import EtoVTree
from util.tool import VTreetoE
import json
data_path = "../data/binary_search/"

class M2Measure:
    def __init__(self,network:Network,p_correct):
        self.topo = network
        self.p = p_correct
        self.init_indm2()
        self.init_m2_true()
        pass


    def measure_m2(self,node1,node2):
        """
        测量m2
        :param node1:
        :param node2:
        :return:
        """
        assert node1 < node2
        num_paths = len(self.topo.getDestinations())
        index = self.indm2[node1-1][node2-1]-1
        true_len = self.m2_true[index]
        pr = np.ones(num_paths - 1) * ((1 - self.p) / (num_paths - 2))
        pr[true_len-1] = self.p
        m2_measured = np.random.choice(list(range(1, num_paths)), 1, replace=True, p=pr)[0]
        return int(m2_measured)

    def measure_m2_all(self):
        measured_m2 = []
        n = len(self.topo.getDestinations())
        for i in range(n):
            for j in range(n):
                if i < j:
                    node1 = i+1
                    node2 = j+1
                    measured_m2.append(self.measure_m2(node1,node2))
        return measured_m2

    def init_indm2(self):
        n = len(self.topo.getDestinations())
        num = 1
        ind_m = np.zeros(shape=(n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if i < j:
                    ind_m[i][j] = num
                    num += 1
        self.indm2 = ind_m

    def init_m2_true(self):
        """
        真实的两两测量值
        :return:
        """
        root_node = self.topo.getSource()
        dest_node = self.topo.getDestinations()
        m2_true = []
        for node1 in dest_node:
            for node2 in dest_node:
                if node1 < node2:
                    path_route1 = nx.shortest_path(self.topo.G, source=root_node, target=node1)
                    path_route2 = nx.shortest_path(self.topo.G, source=root_node, target=node2)
                    shared_path_route = set(path_route1).intersection(set(path_route2))
                    shared_path_len = len(shared_path_route) - 1
                    m2_true.append(shared_path_len)
        self.m2_true = m2_true


class M3Measure:
    def __init__(self, network: Network, p_correct):
        self.topo = network
        self.p = p_correct
        self.m2Measure = M2Measure(network,p_correct)

    def measure_m3(self,node1,node2,node3):
        type = None
        if node1 > node2:
            m2_12 = self.m2Measure.measure_m2(node2,node1)
        else:
            m2_12 = self.m2Measure.measure_m2(node1,node2)
        if node1 > node3:
            m2_13 = self.m2Measure.measure_m2(node3,node1)
        else:
            m2_13 = self.m2Measure.measure_m2(node1,node3)
        if node2 > node3:
            m2_23 = self.m2Measure.measure_m2(node3,node2)
        else:
            m2_23 = self.m2Measure.measure_m2(node2,node3)
        if m2_12 == m2_13 == m2_23:
            type = 0
        elif m2_12 > m2_13 == m2_23:
            type = 1
        elif m2_13 > m2_12 == m2_23:
            type = 2
        elif m2_12 == m2_13 < m2_23:
            type = 3
        else:  # 由于编号规则，type为2的类型不可能，所以type为2以及其他情况，随机type
            type = int(np.random.choice([0,1,3], 1, replace=True, p=[1/3,1/3,1/3])[0])
        return type

    def measure_m3_all(self):
        measured_m3 = []
        num_paths = len(self.topo.getDestinations())
        for iIndex in range(num_paths):
            for jIndex in range(num_paths):
                for kIndex in range(num_paths):
                    if iIndex != jIndex and iIndex != kIndex and jIndex != kIndex:
                        node1 = iIndex+1
                        node2 = jIndex+1
                        node3 = kIndex+1
                        type = self.measure_m3(node1,node2,node3)
                        measured_m3.append(type)
        return measured_m3




def gen1():
    logger = logging.getLogger("data_generator")
    # overlay_node_set, E = ([0, 1, 2, 3, 4], [(0, 5), (1, 5), (2, 6), (3, 6), (4, 6), (5, 6)])
    overlay_node_set, E = ([0,1,2,3,4],[(0, 5), (1, 5), (2, 6), (3, 6), (4, 6), (5, 6)])
    network = Network(overlay_node_set, E, logger=logger)
    network.plot_tree()
    p_correct = 0.6
    m2Measure = M2Measure(network=network,p_correct=p_correct)
    # measure_len = m2Measure.measure_m2(2,3)
    for _ in range(10):
        measured_m2 = m2Measure.measure_m2_all()
        print(measured_m2)

def gen_data_binary_search_p_correct():
    """
    变化测量正确的概率
    :return:
    """
    p_correct_set = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
    times = 100
    logger = logging.getLogger("data_generator")
    for overlay_node_set,tree_vector in topo_zoo_set:
        data_dict = {}
        E = VTreetoE(tree_vector)
        num_path = len(overlay_node_set)-1
        data_dict["probability"] = [str(p) for p in p_correct_set]
        data_dict["tree_vector"] = tree_vector
        data_dict["experiment_times"] = times
        data_dict["num_leaf"] = num_path
        network = Network(overlay_node_set, E, logger=logger)
        # network.plot_tree()
        data_dict["topo_3_data"] = {}
        for p_correct in p_correct_set:
            data_dict["topo_3_data"][str(p_correct)] = []
            for i in range(times):
                m3_generator = M3Measure(network,p_correct=p_correct)
                measured_m3 = m3_generator.measure_m3_all()
                # print(measured_m3)
                data_dict["topo_3_data"][str(p_correct)].append(measured_m3)
        filename = data_path+"p_correct_set_"+str(num_path)+".txt"
        with open(filename,"w") as f:
            json.dump(data_dict,f,indent=4)

def gen_data_binary_search_topo_zoo():
    logger = logging.getLogger("data_generator")
    times = 100
    p_correct = 0.9
    for overlay_nodes,vector_tree in topo_zoo_set:
        data_dict = {}
        data_dict["probability"] = [str(p_correct)]
        data_dict["tree_vector"] = vector_tree
        data_dict["experiment_times"] = times
        data_dict["num_leaf"] = len(overlay_nodes) - 1
        E = VTreetoE(vector_tree)
        network = Network(overlay_nodes, E, logger=logger)
        network.plot_tree()
        data_dict["topo_3_data"] = {}
        data_dict["topo_3_data"][str(p_correct)] = []
        for i in range(times):
            m3_generator = M3Measure(network,p_correct=p_correct)
            measured_m3 = m3_generator.measure_m3_all()
            # print(measured_m3)
            data_dict["topo_3_data"][str(p_correct)].append(measured_m3)
        with open("../data/binary_search/"+str(vector_tree)+".json","w") as f:
            json.dump(data_dict,f,indent=4)





if __name__ == '__main__':
    gen_data_binary_search_p_correct()
    # gen_data_binary_search_topo_zoo()
    # gen1()
    pass