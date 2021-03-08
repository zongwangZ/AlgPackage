#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'run'.py
@ide     : 'PyCharm'
@time    : '2020'-'11'-'13' '18':'52':'34'
@contact : zongwang.zhang@outlook.com
'''
from network import Network
from data_generator.InterferenceMatrixGenerator import InterferenceMatrixGenerator
from algorithm.ILPAlg import ILPAlg
import numpy as np
from algorithm.IdentifyTreeAlg import *
from algorithm.OBAlg import *
import logging
import logging.config
import time
def doSim_ILP():
    """
    采用的是1为根节点，overlay node 自动生成为1到n
    :return:
    """
    # E = [(1, 4), (2, 5), (3, 6), (4, 5), (4, 6), (5, 6)]
    # E = [(1,8),(2,10),(3,12),(4,7),(5,9),(6,11),(7,8),(7,9),(8,10),(9,10),(9,11),(10,12),(11,12)]
    E = [(1, 8), (2, 8), (3, 9), (4, 9), (5, 10), (6, 10), (7, 10), (8, 9), (9, 10)]
    # network = Network(3,E)
    # network = Network(6,E)
    network = Network(7, E)
    generator = InterferenceMatrixGenerator(network)
    interference_matrix = generator.getInterferenceMatrix()
    print(interference_matrix)
    print(np.sum(interference_matrix, 0))
    # solver = ILPAlg(7,3,interference_matrix)
    # solver = ILPAlg(12, 6, interference_matrix)
    # solver = ILPAlg(4, 2, interference_matrix)
    # solver = ILPAlg(8, 4, interference_matrix)
    solver = ILPAlg(10, 7, interference_matrix)
    solver.solve()
    solver.getOutcome()

def doSim_1():
    """
    使用IdentifyTree算法推断
    :return:
    """
    overlay_node_set = [0,1,2,3,4,5,6]
    E = [(0, 7), (1, 7), (2, 8), (3, 8), (4, 9), (5, 9), (6, 9), (7, 8), (8, 9)]
    # E = [(0,6),(1,7),(2,7),(3,8),(4,8),(5,8),(6,7),(6,8)]
    # overlay_node_set = [0,1,2,3,4,5]
    network = Network(overlay_node_set, E)
    generator = InterferenceMatrixGenerator(network)
    interference_matrix = generator.getInterferenceMatrix()
    print(interference_matrix)
    alg = IdentifyTreeAlg(network, overlay_node_set , interference_matrix)
    alg.compute_ed()

def get3(f=0):
    overlay_node_set = None
    E = None
    if f == 0:
        overlay_node_set = [0, 1, 2, 3]
        E = [(0, 4), (1, 4), (2, 4), (3, 4)]
    if f == 1:
        overlay_node_set = [0, 1, 2, 3]
        E = [(0, 4), (1, 4), (2, 5), (3, 5), (4, 5)]
    return overlay_node_set,E

def get4(f=0):
    overlay_node_set = None
    E = None
    if f==0:
        overlay_node_set = [0, 1, 2, 3, 4]
        E = [(0, 5), (1, 5), (2, 6), (3, 6), (4, 6), (5, 6)]
    if f == 1:
        overlay_node_set = [0,1,2,3,4]
        E = [(0,5),(1,6),(2,6),(3,7),(4,7),(5,6),(5,7)]
    return overlay_node_set,E

def get5(f=0):
    overlay_node_set = None
    E = None
    if f == 0:
        overlay_node_set = [0, 1, 2, 3, 4, 5]
        E = [(0, 6), (1, 7), (2, 7), (3, 8), (4, 8), (5, 8), (6, 7), (6, 8)]
        return overlay_node_set, E
    if f == 1:
        overlay_node_set = [0, 1, 2, 3, 4, 5]
        E = [(0, 6), (1, 7), (2, 7), (3, 8), (4, 9), (5, 9), (6, 7), (6, 8),(8,9)]
        return overlay_node_set, E



def doSim_OCCAM():
    logging.config.fileConfig('log_config/logging.conf')
    logger = logging.getLogger('applog')
    logger.info("-----------------------------------"+"start at "+time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())+"-------------------------------")
    import sys
    fname = sys.argv[0]
    with open(fname,"r",encoding="utf-8") as f:
        content = f.read()
        logger.debug(content)
    overlay_node_set, E = get4(1)
    network = Network(overlay_node_set, E,logger=logger)
    alg = OCCAMAlg(network, overlay_node_set,logger=logger)
    alg.true_objective_value()
    alg.solve()
    alg.getOutcome()
    alg.inference()
    # alg.plot_inferred_graph()

def doSim_BI():
    from data_generator.M3Generator import M3Generator
    logging.config.fileConfig('log_config/logging_BI.conf')
    logger = logging.getLogger('applogBI')
    logger.info("-----------------------------------" + "start at " + time.strftime("%Y-%m-%d %H-%M-%S",

                                                                          time.localtime()) + "-------------------------------")
    list1 = [([0,1,2,3,4],[(0, 5), (1, 5), (2, 6), (3, 6), (4, 6), (5, 6)]),
     # ([0,1,2,3,4,5,6],[(0,7),(1,8),(2,8),(3,8),(4,9),(5,10),(6,10),(7,8),(7,9),(9,10)]),
     # ([0,1,2,3,4,5,6,7,8],[(0,9),(1,9),(2,11),(3,11),(4,12),(5,12),(6,12),(7,13),(8,13),(9,10),(9,13),(10,11),(10,12)]),
     # ((0,1,2,3,4,5,6,7,8,9,10),[(0,11),(1,12),(2,13),(3,14),(4,14),(5,14),(6,15),(7,15),(8,16),(9,16),(10,16),(11,12),(11,15),(12,13),(13,14),(15,16)])
    ]
    for (overlay_node_set, E) in list1:
        # overlay_node_set, E = get4(1)
        network = Network(overlay_node_set, E,logger=logger)
        network.plot_tree()
        m3_generator = M3Generator(network,num_time_slots=1000,logger=logger)
        sim_data = m3_generator.getSimData()
        from algorithm.BIHMCAlg import BIHMC
        used = "pystan"
        alg = BIHMC(sim_data,logger,way=used)
        alg.inference()
        alg.get_outcome()

if __name__ == '__main__':
    # doSim_OCCAM()
    # doSim_1()
    doSim_BI()
