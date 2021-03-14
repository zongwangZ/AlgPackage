#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'PathMetricGenerator'.py
@ide     : 'PyCharm'
@time    : '2021'-'01'-'26' '10':'06':'43'
@contact : zongwang.zhang@outlook.com
'''
"""
生成PSM 和 DM
用于OBAlg算法的输入
"""
import numpy as np
import logging
from logging import config
from network import Network
class PathMetricGenerator:
    def __init__(self,topology:Network, r_ns_true=1.0,p_correct=(0.7,0.4),logger=None):
        self.__init_logger(logger)
        self.__topoloy = topology
        self.r_ns_true = r_ns_true  # 网络处于适宜状态概率
        self.p_correct = p_correct  # 分别为网络处于适宜状态 和 不适宜状态下 两两测量正确的概率


    def genPSM(self):
        """
        类似 M3Generator 的方式，生成公共路径长度
        :return:
        """
        num_paths = len(self.__topoloy.getDestinations())
        num_m2 = num_paths * (num_paths - 1) / 2
        num_m2 = int(num_m2)
        m2_measured = np.zeros(shape=(num_m2, ), dtype=int)
        p = None
        if np.random.random() < self.r_ns_true:
            p = self.p_correct[0]
        else:
            p = self.p_correct[1]
        pr = np.ones(num_paths - 1) * ((1 - p) / (num_paths - 2))
        for path in range(num_m2):
            pr_path = np.copy(pr)
            true_path_len = m2_true[path]
            pr_path[true_path_len - 1] = p  # m2测量准确的概率
            m2_measured[path][t] = np.random.choice(list(range(1, num_paths)), 1, replace=True, p=pr_path)

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


