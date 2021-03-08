#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'run_OCCAM'.py
@ide     : 'PyCharm'
@time    : '2021'-'01'-'26' '09':'58':'24'
@contact : zongwang.zhang@outlook.com
'''
import logging
from logging import config
import time
from network import Network
from algorithm.OBAlg import OCCAMAlg

logging.config.fileConfig('log_config/logging.conf')
logger = logging.getLogger('applog')
logger.info("-----------------------------------"+"start at "+time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())+"-------------------------------")
import sys
fname = sys.argv[0]
with open(fname,"r",encoding="utf-8") as f:
    content = f.read()
    logger.debug(content)
E = [(6,1),(0,6),(6,2),(8,3),(7,8),(6,7),(8,4),(7,5)]
overlay_node_set = [0,1,2,3,4,5]
network = Network(overlay_node_set, E,logger)

alg = OCCAMAlg(network, overlay_node_set,logger=logger)
alg.true_objective_value()
alg.solve()
alg.getOutcome()
alg.inference()
# alg.plot_inferred_graph()