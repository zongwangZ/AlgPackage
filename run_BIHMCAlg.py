#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'run_BIHMCAlg'.py
@ide     : 'PyCharm'
@time    : '2021'-'03'-'09' '14':'51':'10'
@contact : zongwang.zhang@outlook.com
'''
"""
1.变化拓扑规模，观察算法ED
2.变化适宜/非适宜下测量正确概率，观察算法的ED
"""
import numpy as np
import time
import logging
from logging import config
from network import Network
from data_generator.M3Generator import M3Generator
import os
def do_sim1():
    """
    不同测量正确概率下，算法推测的正确率
    :return:
    """
    systime = time.strftime("%Y-%m-%d %H-%M-%S",time.localtime())
    filepath = "data/pystan/outcome/rst_p_correct"+systime+".txt"
    file = open(filepath,"a+")
    logging.config.fileConfig('log_config/logging_BI.conf')
    logger = logging.getLogger('applogBI')
    logger.info("-"*10 + "start at " + systime + "-"*10)
    overlay_node_set, E = ([0,1,2,3,4,5],[(0, 6), (1, 7), (2, 7), (3, 8), (4, 8), (5, 8), (6, 7), (6, 8)])

    p_correct_set = np.linspace(1,0.5,6)
    for p_correct in p_correct_set:
        for i in range(10):
            logger.info(f"第{str(i)}次实验")
            logger.info("p_correct:"+str(p_correct))
            network = Network(overlay_node_set, E, logger=logger)
            network.plot_tree()
            m3_generator = M3Generator(network, num_time_slots=100, p_correct=(p_correct, 0.5), logger=logger)
            sim_data = m3_generator.getSimData()
            from algorithm.BIHMCAlg import BIHMC
            used = "pystan"
            alg = BIHMC(sim_data, logger, way=used)
            alg.inference()
            inferred_m2, true_m2 = alg.get_outcome()
            file.write("i"+" "+ str(p_correct)+" "+str(inferred_m2) + " " + str(true_m2)+"\n")
    file.close()


def do_sim2():
    topo_set = []


if __name__ == '__main__':
    do_sim1()