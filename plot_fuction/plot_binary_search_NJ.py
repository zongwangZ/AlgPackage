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
import matplotlib
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# matplotlib.use('Agg')
data_path = "../data/binary_search/xt_result/"
from data.topo_zoo.init import topo_zoo_set
import numpy as np
import os
"""
变化两两测量正确率
"""
def plot_p_correct():
    mean_ed_list = [[]]
    pic_path = data_path+"p_correct_var.pdf"
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
        mean_ed_list[0].append(sum_ed / len(inferred_vector_list))
    plot_general(x=p_correct_set,y_list=mean_ed_list,cdl_style_list=["ro-"],legend_list=["LCPB"],
                 linewidth_list=[3],x_label=u"测量正确的概率",y_label=u"平均编辑距离",
                 pic_path=pic_path)

"""
变化拓扑规模
"""
def plot_topo_zoo():
    pic_path = data_path+"topo_var.pdf"
    num_path_list = []
    mean_ed_list = [[]]
    for overlay_nodes,vector_tree in topo_zoo_set:
        filename = data_path+str(vector_tree)+".json"
        f = open(filename,"r")
        data_dict = json.load(f)
        p_correct = float(data_dict.get("probability")[0])
        VTree = data_dict.get("tree_vector")
        times = data_dict.get("experiment_times")
        inferred_vector_dict = data_dict.get("new_tree_vector")
        num_leaf = len(overlay_nodes)-1
        inferred_vector_list = inferred_vector_dict[str(p_correct)]
        cnt = 0
        sum_ed = 0
        num_path_list.append(num_leaf)
        for inferred_vector in inferred_vector_list:
            E = VTreetoE(VTree)
            inferredE = VTreetoE(inferred_vector)
            ed = calEDbyzss(E, inferredE)
            if ed == 0:
                cnt += 1
            sum_ed += ed
        print("正确率：", cnt / len(inferred_vector_list))
        print("平均编辑距离：", sum_ed / len(inferred_vector_list))
        mean_ed_list[0].append(sum_ed / len(inferred_vector_list))
    plot_general(x=num_path_list,y_list=mean_ed_list,cdl_style_list=["ro-"],legend_list=["LCPB"],
                 linewidth_list=[3],x_label=u"路径数量",y_label=u"平均编辑距离",
                 pic_path=pic_path)

def plot_measure_complexity():
    num_path_list = []
    mean_num_list = [[],[]]
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
        num_path_list.append(num_leaf)
        mean_num_list[0].append(np.mean(num_topo3))
        mean_num_list[1].append(num_leaf*(num_leaf-1)*(num_leaf-2)/6)
        print(np.mean(num_topo3),num_leaf*(num_leaf-1)*(num_leaf-2)/6)
    pic_path = data_path+"measure_complexity.pdf"
    plot_general(x=num_path_list,y_list=mean_num_list,cdl_style_list=["ro-","cs--"],legend_list=["LCPB","TST"],
                 linewidth_list=[3,2],x_label=u"路径数量", y_label=u"平均消耗的三路子拓扑数量",pic_path=pic_path)
def plot_general(x,y_list:list,cdl_style_list,legend_list,linewidth_list,x_label,y_label,pic_path):
    """
    :param cdl_style_list color dot line style
    一般画图函数
    :return:
    """
    num_curve = len(y_list)
    plt.figure(figsize=(5.6, 5.6))
    # plt.rc('text', usetex=True)
    # font = {'family': 'SimHei',
    #         'weight': 'bold',
    #         'size': '16'}
    # plt.rc('font', **font)  # 步骤一（设置字体的更多属性）
    for i in range(num_curve):
        plt.plot(x, y_list[i], cdl_style_list[i], label=legend_list[i], linewidth=linewidth_list[i],
                 markeredgecolor='k', markerfacecoloralt=[0, 0, 0, 0], markersize=10)

    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.yticks(fontsize=16)
    # newxticks=['[5,5]','[5,7]','[5,9]','[5,11]','[5,13]','[5,15]',
    #            '[5,17]','[5,19]','[5,21]','[5,23]','[5,25]']
    plt.xticks(x,fontsize=16)
    plt.xticks(fontsize=16)
    bwith = 2  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.legend(fontsize=16)
    plt.grid(linestyle='--')
    foo_fig = plt.gcf()  # 'get current figure'
    foo_fig.savefig(pic_path, format='pdf', dpi=1000)
    plt.show()

if __name__ == '__main__':
    plot_p_correct()
    plot_topo_zoo()
    plot_measure_complexity()