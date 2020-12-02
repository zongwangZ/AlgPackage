#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'EditDistanceUtil'.py
@ide     : 'PyCharm'
@time    : '2020'-'12'-'01' '21':'57':'04'
@contact : zongwang.zhang@outlook.com
'''
import networkx as nx
from util.tool import *
class EditDistanceUtil:
    def __init__(self):
        pass
    def compute(self,net1:nx.Graph,net2:nx.Graph,root):
        true_edges = nx.dfs_tree(net1, source=root).edges()
        inferred_edges = nx.dfs_tree(net1, source=root).edges()
        inferred_edges = true_edges = numberTopo(inferred_edges,[2,3,4,5,6,7])
        numberTopo(true_edges, [2, 3, 4, 5, 6, 7])
        ed = calEDbyzss(true_edges,inferred_edges,root)
        print(ed)


