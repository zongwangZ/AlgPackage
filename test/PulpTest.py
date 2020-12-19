#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'PulpTest'.py
@ide     : 'PyCharm'
@time    : '2020'-'12'-'14' '13':'55':'14'
@contact : zongwang.zhang@outlook.com
'''
"""
使用IBM的CPlex作为pulp的sovler
方法：
1.指定cplex的位置
path_to_cplex = r'D:\IBM\ILOG\CPLEX_Studio_Community1210\cplex\bin\x64_win64\cplex.exe'
solver = pl.CPLEX_CMD(path=path_to_cplex)
2.配置环境变量，指明cplex的位置
1，2参考https://coin-or.github.io/pulp/guides/how_to_configure_solvers.html
3.将D:\IBM\ILOG\CPLEX_Studio_Community1210\cplex\python\3.7\x64_win64中的cplex文件夹拷贝到
anaconda的Lib\site-packages文件中。

注意clpex临时生成的lp文件，默认放在用户临时目录中，可以使用solver = pl.CPLEX_CMD(keepFiles=True)，存放临时文件到
本文件同一级目录中，可以solver.tmpdir指定目录。
"""
import pulp as pl
from util import Logger

def getCplexSolver():
    """
    获取IBM的CPlex
    :return:
    """
    solver = pl.CPLEX_CMD(keepFiles=True)
    return solver


if __name__ == '__main__':
    solver_list = pl.listSolvers()
    available_solver_list = pl.listSolvers(onlyAvailable=True)
    print(solver_list)
    print(available_solver_list)
    # solver = pl.CPLEX_PY()
    # solver = pl.CPLEX_CMD(keepFiles=True,options = ['epgap = 0.25'])
    # print(solver)
    # path_to_cplex = r'D:\IBM\ILOG\CPLEX_Studio_Community1210\cplex\bin\x64_win64\cplex.exe'
    # solver = pl.CPLEX_CMD(path=path_to_cplex)
    # model = pl.LpProblem("Example", pl.LpMinimize)
    # _var = pl.LpVariable('a')
    # _var2 = pl.LpVariable('a2')
    # model += _var + _var2 == 1
    # result = model.solve(solver)
    # print(result)