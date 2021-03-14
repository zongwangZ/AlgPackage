#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'plot_binary_search'.py
@ide     : 'PyCharm'
@time    : '2021'-'03'-'13' '16':'40':'52'
@contact : zongwang.zhang@outlook.com
'''
"""
用于二分算法的数据分析和画图
"""
from util.tool import VTreetoE
from util.tool import calEDbyzss
import json
from matplotlib import pyplot as plt
data_path = "../data/binary_search/xt_result/"
from data.topo_zoo.init import topo_zoo_set
import numpy as np
"""
变化两两测量正确率
"""
def plot_p_correct():
    filepath = data_path + "p_correct_set.json"
    f = open(filepath,"r")
    data_dict = json.load(f)
    assert isinstance(data_dict,dict)
    p_correct_set = [float(p) for p in data_dict.get("probability")]
    VTree = data_dict.get("tree_vector")
    times = data_dict.get("experiment_times")
    num_leaf = data_dict.get("num_leaf")
    inferred_vector_dict = data_dict.get("new_tree_vector")
    for p in p_correct_set:
        inferred_vector_list = inferred_vector_dict[str(p)]
        cnt = 0
        sum_ed = 0
        for inferred_vector in inferred_vector_list:
            E = VTreetoE(VTree)
            inferredE = VTreetoE(inferred_vector)
            ed = calEDbyzss(E, inferredE)
            if ed == 0:
                cnt += 1
            sum_ed += ed
        print("正确率：",cnt/len(inferred_vector_list))
        print("平均编辑距离：",sum_ed/len(inferred_vector_list))

"""
变化拓扑规模
"""
def plot_topo_zoo():
    for _,vector_tree in topo_zoo_set:
        filename = data_path+str(vector_tree)+".json"
        f = open(filename,"r")
        data_dict = json.load(f)
        p_correct = float(data_dict.get("probability")[0])
        VTree = data_dict.get("tree_vector")
        times = data_dict.get("experiment_times")
        inferred_vector_dict = data_dict.get("new_tree_vector")
        num_leaf = data_dict.get("num_leaf")
        inferred_vector_list = inferred_vector_dict[str(p_correct)]
        cnt = 0
        sum_ed = 0
        for inferred_vector in inferred_vector_list:
            E = VTreetoE(VTree)
            inferredE = VTreetoE(inferred_vector)
            ed = calEDbyzss(E, inferredE)
            if ed == 0:
                cnt += 1
            sum_ed += ed
        print("正确率：", cnt / len(inferred_vector_list))
        print("平均编辑距离：", sum_ed / len(inferred_vector_list))

def plot_measure_complexity():
    for overlay_nodes, vector_tree in topo_zoo_set:
        filename = data_path + str(vector_tree) + ".json"
        f = open(filename, "r")
        data_dict = json.load(f)
        p_correct = float(data_dict.get("probability")[0])
        VTree = data_dict.get("tree_vector")
        times = data_dict.get("experiment_times")
        # num_leaf = data_dict.get("num_leaf")
        num_leaf = len(overlay_nodes)-1
        num_topo3 = data_dict.get("num_topo3").get(str(p_correct))

        print(np.mean(num_topo3),num_leaf*(num_leaf-1)*(num_leaf-2)/6)


if __name__ == '__main__':
    # plot_p_correct()
    # plot_topo_zoo()
    plot_measure_complexity()