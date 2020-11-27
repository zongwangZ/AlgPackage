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
if __name__ == '__main__':
    E = [(1, 4), (2, 5), (3, 6), (4, 5), (4, 6), (5, 6)]
    network = Network(3,E)
    generator = InterferenceMatrixGenerator(network)
    interference_matrix = generator.getInterferenceMatrix()
    print(interference_matrix)
    solver = ILPAlg(6,3,interference_matrix)
    # solver = ILPAlg(4, 2, interference_matrix)
    solver.solve()
    solver.getOutcome()

