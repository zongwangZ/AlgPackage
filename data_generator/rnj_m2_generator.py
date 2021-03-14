#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'rnj_m2_generator'.py
@ide     : 'PyCharm'
@time    : '2021'-'03'-'13' '20':'38':'05'
@contact : zongwang.zhang@outlook.com
'''
from network import Network
from data_generator.binary_search_generator import M2Measure
import logging
def gen_data_rnj_p_correct():
    logger = logging.getLogger("data_generator")
    overlay_node_set, E = ([0, 1, 2, 3, 4, 5], [(0, 6), (7, 1), (7, 2), (8, 3), (8, 4), (8, 5), (6, 7), (6, 8)])
    network = Network(overlay_node_set, E, logger=logger)
    network.plot_tree()
    p_correct = 0.6
    m2Measure = M2Measure(network=network, p_correct=p_correct)
