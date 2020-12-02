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
def doSim_ILP():
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
    # overlay_num = 7
    # E = [(1, 8), (2, 8), (3, 9), (4, 9), (5, 10), (6, 10), (7, 10), (8, 9), (9, 10)]
    overlay_num = 6
    E = [(1,7),(2,8),(3,8),(4,9),(5,9),(6,9),(7,8),(7,9)]
    network = Network(overlay_num, E)
    generator = InterferenceMatrixGenerator(network)
    interference_matrix = generator.getInterferenceMatrix()
    print(interference_matrix)
    alg = IdentifyTreeAlg(network,overlay_num,interference_matrix)
    alg.compute_ed()



if __name__ == '__main__':
    doSim_1()

