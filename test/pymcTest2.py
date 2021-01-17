#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'pymcTest2'.py
@ide     : 'PyCharm'
@time    : '2020'-'12'-'30' '16':'24':'03'
@contact : zongwang.zhang@outlook.com
'''
"""
自定义似然函数 test1
https://discourse.pymc.io/t/how-to-set-up-a-custom-likelihood-function-for-two-variables/906/5
"""
import pymc3 as pm
import random
import numpy as np
import theano.tensor as tt
# Underlying parameters for prior 1
sigma_1 = 0.49786792462
mu_1 = 0.510236734833

# Underlying parameters for prior 2
sigma_2 = 0.62667474617
mu_2 = 0.169577686173

# Physical properties
# Unit as default(KG)
k_1 = 29.7 * 10 ** (6)
k_2 = 29.7 * 10 ** (6)

m_1 = 16.5 * 10 ** (3)
m_2 = 16.1 * 10 ** (3)

# Frequency observations
fm_1 = 3.13
fm_2 = 9.83

with pm.Model() as model:
    theta_1 = pm.Lognormal('theta_1', mu=mu_1, sd=sigma_1)
    theta_2 = pm.Lognormal('theta_2', mu=mu_2, sd=sigma_2)


    def likelihood(theta_1, theta_2):
        def obs(f):
            a = m_1 * m_2
            b = m_1 * k_2 * theta_2 + m_2 * k_2 * theta_2 + m_2 * k_1 * theta_1
            c = theta_1 * theta_2 * k_1 * k_2

            # Contour parameter
            i0 = 9

            f_1_square = (b - tt.sqrt(b ** 2 - 4 * a * c)) / (2 * a) / (4 * np.pi ** 2)
            f_2_square = (b + tt.sqrt(b ** 2 - 4 * a * c)) / (2 * a) / (4 * np.pi ** 2)

            return tt.exp(-2 ** (i0 - 2) * ((f_1_square / f[0] ** 2 - 1) ** 2 +
                                            (f_2_square / f[1] ** 2 - 1) ** 2))

        return obs

    like = pm.DensityDist('like', likelihood(theta_1,theta_2), observed=[fm_1,fm_2])
    start = pm.find_MAP(model = model)
    step = pm.Metropolis()
    trace = pm.sample(10000,step=step,start=start)

pm.traceplot(trace)

