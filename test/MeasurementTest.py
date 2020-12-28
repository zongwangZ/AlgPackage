#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'MeasurementTest'.py
@ide     : 'PyCharm'
@time    : '2020'-'12'-'26' '19':'56':'37'
@contact : zongwang.zhang@outlook.com
'''

"""
测试两种测量方法对三路子拓扑数据的影响
"""
import random
import numpy as np
from matplotlib import pyplot as plt

n = 10000  # 重复实验的次数

def fun1(alpha):
    """
    根据alpha生成链路长度，2，1，0概率分别为(1-alpha)/2,alpha,(1-alpha)/2
    :return:
    """
    p = [alpha,(1+alpha)/2,1]
    r = random.random()
    if p[0] > r >= 0:
        return 1
    if p[1] > r >= p[0]:
        return 0
    if p[2] > r >= p[1]:
        return 2

def sim1():
    # 1.随alpha的变化，acc的变化
    alphas = [i/100 for i in range(70,101)]
    D = [1, 2, 3]
    acc_0_sim_r = []
    acc_1_sim_r = []
    acc_0_pair_r = []
    acc_1_pair_r = []
    for alpha in alphas:
        # 同时测量
        acc_0_sim = []
        acc_1_sim = []
        for i in range(n):
            acc_0_sim.append(1)
            link_len = fun1(alpha)
            if link_len == 0:
                acc_1_sim.append(0)
            if link_len == 1 or 2:
                acc_1_sim.append(1)

        # 两两测量
        acc_0_pair = []
        acc_1_pair = []
        for i in range(n):
            path12 = fun1(alpha)
            path13 = fun1(alpha)
            path23 = fun1(alpha)
            if path12 == path13 == path23:
                acc_0_pair.append(1)
            else:
                acc_0_pair.append(0)
            path12 = fun1(alpha) + fun1(alpha)
            path13 = fun1(alpha)
            path23 = fun1(alpha)
            if path12 > path13 == path23:
                acc_1_pair.append(1)
            else:
                acc_1_pair.append(0)
        acc_0_sim_r.append(np.mean(np.array(acc_0_sim)))
        acc_1_sim_r.append(np.mean(np.array(acc_1_sim)))
        acc_0_pair_r.append(np.mean(np.array(acc_0_pair)))
        acc_1_pair_r.append(np.mean(np.array(acc_1_pair)))

    plt.subplot()
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(alphas,acc_0_sim_r, "^-",label="同时测量，拓扑类型0")
    plt.plot(alphas,acc_1_sim_r,"cs-",label="同时测量，拓扑类型1")
    plt.plot(alphas,acc_0_pair_r,".-",label="两两测量，拓扑类型0")
    plt.plot(alphas,acc_1_pair_r,"o-",label="两两测量，拓扑类型1")
    plt.xlabel("alpha", fontsize=16)
    plt.ylabel("acc", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(linestyle='--')
    plt.show()

def sim2():
    alpha = 0.8
    share_paths = [1,2,3,4,5,6]
    D = [1, 2, 3]
    acc_0_sim_r = []
    acc_1_sim_r = []
    acc_0_pair_r = []
    acc_1_pair_r = []
    for share_path in share_paths:
        # 同时测量
        acc_0_sim = []
        acc_1_sim = []
        for i in range(n):
            acc_0_sim.append(1)
            link_len = 0
            for j in range(share_path):
                link_len += fun1(alpha)
            if link_len == 0:
                acc_1_sim.append(0)
            else:
                acc_1_sim.append(1)
        # 两两测量
        acc_0_pair = []
        acc_1_pair = []
        for i in range(n):
            path12 = 0
            for j in range(share_path):
                path12 += fun1(alpha)
            path13 = 0
            for j in range(share_path):
                path13 += fun1(alpha)
            path23 = 0
            for j in range(share_path):
                path23 += fun1(alpha)
            if path12 == path13 == path23:
                acc_0_pair.append(1)
            else:
                acc_0_pair.append(0)

            path12 = fun1(alpha)
            for j in range(share_path):
                path12 += fun1(alpha)
            path13 = fun1(alpha)
            path23 = fun1(alpha)
            if path12 > path13 == path23:
                acc_1_pair.append(1)
            else:
                acc_1_pair.append(0)
        acc_0_sim_r.append(np.mean(np.array(acc_0_sim)))
        acc_1_sim_r.append(np.mean(np.array(acc_1_sim)))
        acc_0_pair_r.append(np.mean(np.array(acc_0_pair)))
        acc_1_pair_r.append(np.mean(np.array(acc_1_pair)))
    plt.subplot()
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(share_paths,acc_0_sim_r, "^-",label="同时测量，拓扑类型0")
    plt.plot(share_paths,acc_1_sim_r,"cs-",label="同时测量，拓扑类型1")
    plt.plot(share_paths,acc_0_pair_r,".-",label="两两测量，拓扑类型0")
    plt.plot(share_paths,acc_1_pair_r,"o-",label="两两测量，拓扑类型1")
    plt.xlabel("share_paths", fontsize=16)
    plt.ylabel("acc", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(linestyle='--')
    plt.title("alpha="+str(alpha))
    plt.show()

if __name__ == '__main__':
    sim1()
    sim2()