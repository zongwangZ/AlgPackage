#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'pymcTest3'.py
@ide     : 'PyCharm'
@time    : '2021'-'01'-'10' '16':'18':'55'
@contact : zongwang.zhang@outlook.com
'''

"""
自定义似然函数 test1
"""

import pymc3 as pm
import random
model = pm.Model()
with model:
    p_prior = pm.Uniform("p_prior",0.8,1.0)
    # def likelihood(p_prior):
    #     if random.random() < p_prior:
    #         p = 0.9
    #     else:
    #         p = 0.5
    #
    #
    # pm.DensityDist("likelihood",likelihood)
    likelihood = pm.Bernoulli(name="test",p=p_prior)


