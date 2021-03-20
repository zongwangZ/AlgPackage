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
from algorithm.BIHMCAlg import BIHMC
import os
from util.tool import VTreetoE
def do_sim_p_correct():
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
    times = 1
    p_correct_set = np.linspace(1,0.5,6)
    for p_correct in p_correct_set:
        for i in range(times):
            logger.info(f"第{str(i)}次实验")
            logger.info("p_correct:"+str(p_correct))
            network = Network(overlay_node_set, E, logger=logger)
            # network.plot_tree()
            m3_generator = M3Generator(network, num_time_slots=1, p_correct=(p_correct, 0.5), logger=logger)
            sim_data = m3_generator.getSimData()
            from algorithm.BIHMCAlg import BIHMC
            used = "pystan"
            alg = BIHMC(sim_data, logger, way=used)
            alg.inference()
            inferred_m2, true_m2 = alg.get_outcome()
            file.write(str(i)+" "+ str(p_correct)+" "+str(inferred_m2) + " " + str(true_m2)+"\n")
    file.close()


def do_sim_topo_var():
    from data.topo_zoo.init import topo_zoo_set
    systime = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    filepath = "data/pystan/outcome/rst_topo_var" + systime + ".txt"
    with open(filepath, "a+") as file:
        logging.config.fileConfig('log_config/logging_BI.conf')
        logger = logging.getLogger('applogBI')
        logger.info("-" * 10 + "start at " + systime + "-" * 10)
        times = 1
        p_correct = 0.9
        for overlay_node_set, vecotr_tree in topo_zoo_set:
            E = VTreetoE(vecotr_tree)
            # if len(overlay_node_set)-1 < 6:
            #     continue
            for i in range(times):
                logger.info(f"第{str(i)}次实验")
                logger.info("p_correct:" + str(p_correct))
                network = Network(overlay_node_set, E, logger=logger)
                # network.plot_tree()
                m3_generator = M3Generator(network, num_time_slots=1,r_ns_true=1, p_correct=(p_correct, 0.5), logger=logger)
                sim_data = m3_generator.getSimData()
                used = "pystan"
                alg = BIHMC(sim_data, logger, way=used)
                alg.inference()
                inferred_m2, true_m2 = alg.get_outcome()
                file.write(str(i) + " " + str(vecotr_tree) + " " + str(inferred_m2) + " " + str(true_m2) + "\n")


if __name__ == '__main__':
    # do_sim_topo_var()
    do_sim_p_correct()