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
from log_config.Logger import Logger
from data.topo_zoo.init import topo_zoo_set
used = "pystan"
def verify_algorithm():
    """
    验证算法在输入完备的情况下，算法推断正确
    :return:
    """
    # 关键参数
    p = 0.9
    num_time_slot = 400
    iteration_steps = 5000
    debug = 0

    logger = Logger("log_BI","verify_algorithm",if_file=False,if_console=False)
    overlay_node_set, E = ([0, 1, 2, 3, 4], [(6, 1), (6, 2), (7, 3), (7, 4), (0, 5), (5, 6), (5, 7)])
    # overlay_node_set, E = ([0,1,2,3,4,5],[(0, 6), (1, 7), (2, 7), (3, 8), (4, 8), (5, 8), (6, 7), (6, 8)])
    network = Network(overlay_node_set, E, logger=logger)
    m3_generator = M3Generator(network, num_time_slots=num_time_slot, p_correct=(p, 0.5), logger=logger)
    sim_data = m3_generator.getSimData()
    alg = BIHMC(sim_data, logger, debug=debug,iteration_steps=iteration_steps,way=used)
    alg.inference()
    inferred_m2, true_m2 = alg.get_outcome(ifplot=True)
    print(inferred_m2)
    print([round(i) for i in inferred_m2])
    print(true_m2)


def do_sim_p_correct():
    """
    不同测量正确概率下，算法推测的正确率
    :return:
    """
    systime = time.strftime("%Y-%m-%d %H-%M-%S",time.localtime())
    filepath = "data/pystan/outcome/rst_p_correct"+systime+".txt"
    with open(filepath,"a+") as file:
        logger = Logger("log_BI", "p_correct")
        logger.info("-"*10 + "start at " + systime + "-"*10)
        # overlay_node_set,E = ([0,1,2,3,4],[(6,1),(6,2),(7,3),(7,4),(0,5),(5,6),(5,7)])
        overlay_node_set, E = ([0,1,2,3,4,5],[(0, 6), (1, 7), (2, 7), (3, 8), (4, 8), (5, 8), (6, 7), (6, 8)])
        times = 1
        # p_correct_set = [1]
        p_correct_set = np.linspace(1,0.5,6)
        for p_correct in p_correct_set:
            for i in range(times):
                logger.info(f"第{str(i)}次实验")
                logger.info("p_correct:"+str(p_correct))
                network = Network(overlay_node_set, E, logger=logger)
                # network.plot_tree()
                m3_generator = M3Generator(network, num_time_slots=1, p_correct=(p_correct, 0.5), logger=logger)
                sim_data = m3_generator.getSimData()
                alg = BIHMC(sim_data, iteration_steps=100,logger=logger, way=used)
                alg.inference()
                inferred_m2, true_m2 = alg.get_outcome()
                file.write(str(i)+" "+ str(p_correct)+" "+str(inferred_m2) + " " + str(true_m2)+"\n")
        file.close()


def do_sim_topo_var():
    # 关键参数
    times = 1
    p_correct = 0.9
    r_ns_true = 1
    num_time_slots = 1
    iteration_steps = 100
    used = "pystan"
    systime = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    filepath = "data/pystan/outcome/rst_topo_var" + systime + ".txt"
    logger = Logger("log_BI", "p_correct")
    logger.info("p_correct: " + str(p_correct)+" 重复实验次数: "
                +str(times)+" 网络良好概率: "+str(r_ns_true)+" 测量次数: "+str(num_time_slots)
                +" 马尔科夫链迭代次数: "+str(iteration_steps))
    with open(filepath, "a+") as file:
        for overlay_node_set, vecotr_tree in topo_zoo_set:
            E = VTreetoE(vecotr_tree)
            for i in range(times):
                logger.info(f"第{str(i)}次实验")
                network = Network(overlay_node_set, E, logger=logger)
                # network.plot_tree()
                m3_generator = M3Generator(network, num_time_slots=num_time_slots,
                                           r_ns_true=r_ns_true, p_correct=(p_correct, 0.5), logger=logger)
                sim_data = m3_generator.getSimData()
                alg = BIHMC(sim_data, iteration_steps=iteration_steps,logger=logger, way=used)
                alg.inference()
                inferred_m2, true_m2 = alg.get_outcome()
                inferred_m2 = list(inferred_m2)
                file.write(str(i) + "|" + str(vecotr_tree) + "|" + str(inferred_m2) + "|" + str(true_m2) + "\n")


if __name__ == '__main__':
    # do_sim_topo_var()
    # do_sim_p_correct()
    # verify_algorithm()
    pass