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
import pulp as pl

if __name__ == '__main__':
    # solver_list = pl.listSolvers()
    # available_solver_list = pl.listSolvers(onlyAvailable=True)
    # print(solver_list)
    # print(available_solver_list)
    # solver = pl.getSolver('CPLEX_CMD')
    # print(solver)
    path_to_cplex = r'D:\IBM\ILOG\CPLEX_Studio_Community1210\cplex\bin\x64_win64\cplex.exe'
    solver = pl.CPLEX_CMD(path=path_to_cplex)
    model = pl.LpProblem("Example", pl.LpMinimize)
    _var = pl.LpVariable('a')
    _var2 = pl.LpVariable('a2')
    model += _var + _var2 == 1
    result = model.solve(solver)
    print(result)