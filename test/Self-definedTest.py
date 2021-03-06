#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'Self-definedTest'.py
@ide     : 'PyCharm'
@time    : '2021'-'01'-'07' '15':'20':'29'
@contact : zongwang.zhang@outlook.com
'''
"""
自定义函数 pymc3
"""
import pymc3 as pm
import numpy as np
import math
import theano.tensor as tt


def getPrCorrect(netStatus: np.ndarray, pr_network, T, prc):
    """
    对应着R实现中的getPrCorrect
    获得共享路径 m2 正确测量的概率
    :param netStatus:
    :param pr_network:
    :param T:
    :param prc:
    :return:
    """
    pr = np.zeros(T)
    for t in range(T):
        if netStatus[t] <= pr_network:
            pr[t] = prc[0]
        else:
            pr[t] = prc[1]
    return pr


def getThetaM2(sharedPathLength, p_correct, N, T):
    """
    对应着R实现中的getThetaM2
    获得 m2 的概率分布
    :param sharedPathLength:
    :param p_correct:
    :param N:
    :param T:
    :return:
    """
    theta_m2 = np.zeros(shape=(T, N * (N - 1) / 2, N - 1), dtype=int)
    l_true = None
    for t in range(T):
        for n in range(N * (N - 1) / 2):
            l_true = round(sharedPathLength[n])
            for l in range(1, N):  # 公共路径长度
                if l_true - 1 < l <= l_true:
                    theta_m2[t, n, l - 1] = p_correct[t]
                else:
                    theta_m2[t, n, l - 1] = (1 - p_correct[t]) / (N - 2)
    return theta_m2


def getThetaM3(theta_m2, ind_m2, ind_m3, N, T):
    """
    对应R实现中的getThetaM3
    对应论文中的公式(8)-(12)
    获得 m3 的概率分布
    :param theta_m2:
    :param ind_m2:
    :param ind_m3:
    :param N:
    :param T:
    :return:
    """
    theta_m3 = np.zeros(shape=(T, N * (N - 1) * (N - 2) / 6, 5), dtype=int)
    for t in range(T):
        for iIndex in range(N):
            for jIndex in range(N):
                for kIndex in range(N):
                    if iIndex < jIndex < kIndex:
                        ijk = ind_m3[iIndex][jIndex][kIndex] - 1
                        ij = ind_m2[iIndex][jIndex] - 1
                        ik = ind_m2[iIndex][kIndex] - 1
                        jk = ind_m2[jIndex][kIndex] - 1
                        for l in range(1, N):
                            theta_m3[t, ijk, 0] += theta_m2[t, ij, l - 1] * theta_m2[t, ik, l - 1] * theta_m2[
                                t, jk, l - 1]
                            if l >= 2:
                                for l_less in range(1, l):
                                    theta_m3[t, ijk, 1] += theta_m2[t, ij, l - 1] * theta_m2[t, ik, l_less - 1] * \
                                                           theta_m2[
                                                               t, jk, l_less - 1]
                                    theta_m3[t, ijk, 2] += theta_m2[t, ij, l_less - 1] * theta_m2[t, ik, l - 1] * \
                                                           theta_m2[
                                                               t, jk, l_less - 1]
                                    theta_m3[t, ijk, 3] += theta_m2[t, ij, l_less - 1] * theta_m2[t, ik, l_less - 1] * \
                                                           theta_m2[
                                                               t, jk, l - 1]
                        theta_m3[t, ijk, 4] = 1
                        for topo in range(4):  # m3测量失败的概率
                            theta_m3[t, ijk, 4] += -1 * theta_m3[t, ijk, topo]
    return theta_m3


def measure_m2_lpdf(y, theta, num_u):
    """
    对应R实现中的measure_m2_lpmf函数
    m2 概率分布
    :param y:
    :param theta:
    :param num_u:
    :return:
    """
    prob = 0.0
    for i in range(1, num_u + 1):
        if y == i:
            prob = theta[i - 1]
    return math.log(prob)


def measure_m3_lpmf(y, theta):
    """
    对应R实现中的measure_m3_lpmf函数
    m3 概率分布
    :param y:
    :param theta:
    :return:
    """
    prob = 0.0
    for topo in range(4):
        if y == topo:
            prob = theta[topo]
    return math.log(prob)



if __name__ == '__main__':
    N = 4
    T = 100
    prc = [0.9,0.5]
    ind_m2 = get
    with pm.Model():
        def likelihood(net_status, r_ns,m2):
            p = getPrCorrect(net_status,r_ns,T,prc)
            theta_m2 = getThetaM2(m2,p,N,T)
            theta_m3 = getThetaM3(theta_m2,)

        m2 = pm.DiscreteUniform("m2",lower=1,upper=N-1,shape=(int(N*(N-1)/2)))
        # print(m2.random())
        r_ns = pm.Uniform("r_ns",lower=1.0,upper=1.0)
        # print(r_ns.random())
        net_status = pm.Uniform("net_status",lower=0,upper=1,shape=(T))
        getPrCorrect_l = pm.Potential("getPrCorrect",getPrCorrect(net_status,r_ns,T,prc))
        print(getPrCorrect_l)

        # pm.Potential("likelihood",measure_m3_lpmf())
