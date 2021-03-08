#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'run_IdentifyTree'.py
@ide     : 'PyCharm'
@time    : '2021'-'01'-'24' '09':'39':'33'
@contact : zongwang.zhang@outlook.com
'''
from network import Network
from data_generator.InterferenceMatrixGenerator import InterferenceMatrixGenerator
from algorithm.IdentifyTreeAlg import IdentifyTreeAlg
# 1. 导入数据集
p_correct = 0.9
E = [(6,1),(0,6),(6,2),(8,3),(7,8),(6,7),(8,4),(7,5)]
overlay_node_set = [0,1,2,3,4,5]
network = Network(overlay_node_set, E)
generator = InterferenceMatrixGenerator(network, p_correct)
interference_matrix = generator.getInterferenceMatrix()
print(interference_matrix)
alg = IdentifyTreeAlg(network, overlay_node_set , interference_matrix)
alg.inference()
alg.compute_ed()