#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'plot_BI'.py
@ide     : 'PyCharm'
@time    : '2021'-'03'-'23' '09':'52':'54'
@contact : zongwang.zhang@outlook.com
'''
"""
关于贝叶斯拓扑识别的画图函数
"""
data_path = "../data/pystan/outcome/"
import ast

def plot_1():
    file_path = data_path + "rst_topo_var2021-03-20 00-07-39.txt"
    inferred_m2_set = []
    true_m2_set = []
    with open(file_path,"r",encoding="utf-8") as file:
        line = file.readline()
        while line:
            _, line = line.split("[",1)
            _, line = line.split("[",1)
            inferred_m2_str,true_m2_str = line.split("[",1)
            line = file.readline()

def plot_converge():
    """

    :return:
    """
    pass


if __name__ == '__main__':
    plot_1()
    pass