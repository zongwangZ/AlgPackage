#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'pystanTest'.py
@ide     : 'PyCharm'
@time    : '2021'-'01'-'11' '15':'36':'28'
@contact : zongwang.zhang@outlook.com
'''

import pystan
model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
model = pystan.StanModel(model_code=model_code)
y = model.sampling().extract()['y']
y.mean()  # with luck the result will be near 0