#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'ILPTest'.py
@ide     : 'PyCharm'
@time    : '2020'-'11'-'13' '21':'02':'53'
@contact : zongwang.zhang@outlook.com
'''

'''
objective:
min(0.013*x1+0.008*x2+0.01*x3+0.002*x4+0.005*x5+0.001*x6)
 
s.t.
x1,x2,x3,x4,x5,x6 >= 0
0.100*x1+0.200*x2+0.150*x3+0.000*x4+0.040*x5+0.000*x6 >= 8.0
0.080*x1+0.100*x2+0.110*x3+0.010*x4+0.010*x5+0.000*x6 >= 6.0
0.001*x1+0.005*x2+0.003*x3+0.100*x4+0.150*x5+0.000*x6 <= 2.0
0.002*x1+0.005*x2+0.007*x3+0.002*x4+0.008*x5+0.000*x6 <= 2.0

使用pulp工具包解决ILP问题
'''
from pulp import *

Ingredients = ['CHICKEN', 'BEEF', 'MUTTON', 'RICE', 'WHEAT', 'GEL']
costs = {'CHICKEN': 0.013,
         'BEEF': 0.008,
         'MUTTON': 0.010,
         'RICE': 0.002,
         'WHEAT': 0.005,
         'GEL': 0.001}

proteinPercent = {'CHICKEN': 0.100,
                  'BEEF': 0.200,
                  'MUTTON': 0.150,
                  'RICE': 0.000,
                  'WHEAT': 0.040,
                  'GEL': 0.000}

fatPercent = {'CHICKEN': 0.080,
              'BEEF': 0.100,
              'MUTTON': 0.110,
              'RICE': 0.010,
              'WHEAT': 0.010,
              'GEL': 0.000}

fibrePercent = {'CHICKEN': 0.001,
                'BEEF': 0.005,
                'MUTTON': 0.003,
                'RICE': 0.100,
                'WHEAT': 0.150,
                'GEL': 0.000}

saltPercent = {'CHICKEN': 0.002,
               'BEEF': 0.005,
               'MUTTON': 0.007,
               'RICE': 0.002,
               'WHEAT': 0.008,
               'GEL': 0.000}

#创建问题实例，求最小极值
prob = LpProblem("The Whiskas Problem", LpMinimize)

# 构建Lp变量字典，变量名以Ingr开头，如Ingr_CHICKEN，下界是0
ingredient_vars = LpVariable.dicts("Ingr", Ingredients, 0)

# 添加目标方程
prob += lpSum([costs[i] * ingredient_vars[i] for i in Ingredients])

# 添加约束条件
prob += lpSum([ingredient_vars[i] for i in Ingredients]) == 100
prob += lpSum([proteinPercent[i] * ingredient_vars[i] for i in Ingredients]) >= 8.0
prob += lpSum([fatPercent[i] * ingredient_vars[i] for i in Ingredients]) >= 6.0
prob += lpSum([fibrePercent[i] * ingredient_vars[i] for i in Ingredients]) <= 2.0
prob += lpSum([saltPercent[i] * ingredient_vars[i] for i in Ingredients]) <= 0.4

# 求解
prob.solve()
# 查看解的状态
print("Status:", LpStatus[prob.status])
# 查看解
for v in prob.variables():
    print(v.name, "=", v.varValue)
# 另外一种查看解的方式
# for i in Ingredients:
#   print(ingredient_vars[i],"=",ingredient_vars[i].value())

