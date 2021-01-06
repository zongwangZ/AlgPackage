#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'M3Measure'.py
@ide     : 'PyCharm'
@time    : '2021'-'01'-'05' '10':'42':'30'
@contact : zongwang.zhang@outlook.com
'''
"""
测量m3
"""
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import logging
from logging import config
from network import Network
import networkx as nx
class M3Generator:
    def __init__(self,topology:Network, num_time_slots=100, r_ns_true=1.0, p_correct=(0.9, 0.1), logger=None):
        self.__init_logger(logger)
        self.__topoloy = topology
        self.__init_params_R(num_time_slots,r_ns_true,p_correct) # 定义参数，对应R实现中的model_par

    def __init_params_R(self,num_time_slots,r_ns_true,p_correct):
        root_node = self.__topoloy.getSource()
        dest_node = self.__topoloy.getDestinations()
        num_paths = len(dest_node)
        # 计算真实公共路径长度
        m2_true = []
        for node1 in dest_node:
            for node2 in dest_node:
                if node1 < node2:
                    path_route1 = nx.shortest_path(self.__topoloy.G,source=root_node,target=node1)
                    path_route2 = nx.shortest_path(self.__topoloy.G,source=root_node,target=node2)
                    shared_path_route = set(path_route1).intersection(set(path_route2))
                    shared_path_len = len(shared_path_route)-1
                    m2_true.append(shared_path_len)
        ind_m2 = self.__construct_index_matrix_R(num_paths,2)
        ind_m3 = self.__construct_index_matrix_R(num_paths,3)
        self.params = {
            "num_paths":num_paths, # 总共的测量路径数;
            "num_time_slots":num_time_slots, # 总共的测量时隙数;
            "m2_true":m2_true, # 真实的共享路径长度;
            "r_ns_true":r_ns_true, # 网络处于"适宜状态"的概率;
            "p_correct":p_correct,  # 共享路径在网络"适宜"/"非适宜"状态下被正确测量的概率;
            "ind_m2":ind_m2, # 共享路径索引 矩阵式 -> 向量式;
            "ind_m3":ind_m3, # 三路子拓扑索引 矩阵式 -> 向量式.
        }
        self.__logger.info("parameters:",str(self.params))

    def __construct_index_matrix_R(self,n,mDim=2):
        """
        构建index 矩阵，对应着R实现中的index_matrix_no函数
        :return:
        """
        ind_m = None
        if mDim == 2:
            num = 1
            ind_m = np.zeros(shape=(n,n),dtype=int)
            for i in range(n):
                for j in range(n):
                    if i < j:
                        ind_m[i][j] = num
                        num += 1
        elif mDim == 3:
            num = 1
            ind_m = np.zeros(shape=(n,n,n),dtype=int)
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        if i < j < k:
                            ind_m[i][j][k] = num
                            num += 1
        return ind_m


    def __generateM2(self):
        """
        生成m2,从R转过来,对应data_from_sim中的前半部分
        :return:
        """
        num_paths = self.params.get("num_paths")
        num_m2 = num_paths*(num_paths-1)/2
        num_m2 = int(num_m2)
        num_time_slots = self.params.get("num_time_slots")
        m2_measured = np.zeros(shape=(num_m2,num_time_slots),dtype=int)
        r_ns_true = self.params.get("r_ns_true")
        p_correct = self.params.get("p_correct")
        m2_true = self.params.get("m2_true")

        for t in range(num_time_slots):
            if random.random() < r_ns_true:
                p = p_correct[0]
            else:
                p = p_correct[1]

            # pr = np.arange(1,num_paths) * ((1-p)/(num_paths-2))
            pr = np.ones(num_paths-1) * ((1-p)/(num_paths-2))
            for path in range(num_m2):
                pr_path = np.copy(pr)
                true_path_len = m2_true[path]
                pr_path[true_path_len-1] = p  # m2测量准确的概率
                m2_measured[path][t] = np.random.choice(list(range(1,num_paths)),1,replace=True,p=pr_path)

        return m2_measured




    def generateM3(self,measure_way=0):
        """
        生成m3的观测值
        1.同时测量
        2.根据两两测量得到三路测量
        :return:
        """

        if measure_way == 0:
            m2_measured = self.__generateM2()
            num_paths = self.params.get("num_paths")
            num_m3 = num_paths * (num_paths - 1) * (num_paths - 2) / 6
            num_m3 = int(num_m3)
            num_time_slots = self.params.get("num_time_slots")
            m3_measured = np.zeros(shape=(num_m3, num_time_slots), dtype=int)
            ind_m2 = self.params.get("ind_m2")
            ind_m3 = self.params.get("ind_m3")
            for t in range(num_time_slots):
                for iIndex in range(num_paths):
                    for jIndex in range(num_paths):
                        for kIndex in range(num_paths):
                            if iIndex < jIndex < kIndex:
                                ijk = ind_m3[iIndex][jIndex][kIndex]-1
                                ij = ind_m2[iIndex][jIndex]-1
                                ik = ind_m2[iIndex][kIndex]-1
                                jk = ind_m2[jIndex][kIndex]-1
                                if m2_measured[ij][t] == m2_measured[ik][t]  == m2_measured[jk][t]:
                                    m3_measured[ijk][t] = 0
                                elif m2_measured[ij, t] > m2_measured[ik, t] == m2_measured[jk, t]:
                                    m3_measured[ijk][t] = 1
                                elif m2_measured[ik, t] > m2_measured[ij, t] == m2_measured[jk, t]:
                                    m3_measured[ijk, t] = 2
                                elif m2_measured[jk, t] > m2_measured[ij, t] == m2_measured[ik, t]:
                                    m3_measured[ijk, t] = 3
                                else:
                                    m3_measured[ijk, t] = 4






    def __init_logger(self,logger):
        """
        初始化logger
        :return:
        """
        if logger is not None:
            self.__logger = logger
        else:
            logging.config.fileConfig('util/logging.conf')
            self.__logger = logging.getLogger('applog')