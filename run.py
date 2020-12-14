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

def doSim_OCCAM():
    # overlay_node_set = [0, 1, 2, 3, 4, 5, 6]
    # E = [(0, 7), (1, 7), (2, 8), (3, 8), (4, 9), (5, 9), (6, 9), (7, 8), (8, 9)]
    # overlay_node_set = [0,1,2,3,4,5]
    # E = [(0,6),(1,6),(2,6),(3,8),(4,8),(5,7),(6,7),(7,8)]
    overlay_node_set = [0,1,2,3]
    E = [(0,4),(1,4),(2,4),(3,4)]
    # E = [(0,4),(1,4),(2,5),(3,5),(4,5)]
    network = Network(overlay_node_set, E)
    alg = OCCAMAlg(network, overlay_node_set)
    alg.getOutcome()
    alg.solve()
    alg.true_objective_value()
    alg.getOutcome()
    # alg.inference()
    # alg.plot_inferred_graph()

if __name__ == '__main__':
    doSim_OCCAM()

