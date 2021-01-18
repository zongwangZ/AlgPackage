#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'gen_topo_zoo'.py
@ide     : 'PyCharm'
@time    : '2021'-'01'-'17' '19':'41':'57'
@contact : zongwang.zhang@outlook.com
'''

import ast
import util.tool as tool

source_file = "graph_info.txt"
topo_zoo_list = []
with open(source_file,"r") as f:
    line = f.readline()
    while line:
        index = line.index("[")
        fake_vtree = ast.literal_eval(line[index:])
        fake_E = tool.VTreetoE(fake_vtree)
        fake_leafnodes = tool.getLeafNodes(fake_vtree)

        # real_leafnodes = list(range(1,len(fake_leafnodes)+1))
        real_E = tool.numberTopo(fake_E,fake_leafnodes)
        real_vtree = tool.EtoVTree(real_E)
        D = tool.getChildren(real_E,parent=0)
        N = len(D)
        topo_zoo_list.append(dict(
            vtree=real_vtree,
            E=real_E,
            D=D,
            N=N
        ))
print(topo_zoo_list)