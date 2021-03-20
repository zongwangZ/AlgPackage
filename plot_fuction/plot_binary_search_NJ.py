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
data_path_binary_search = "../data/binary_search/xt_result/"
from data.topo_zoo.init import topo_zoo_set
import numpy as np
import os
"""
变化两两测量正确率
"""
def plot_p_correct(ifplot=False):
    mean_ed_list = [[]]
    pic_path = data_path_binary_search+"p_correct_var.pdf"
    filepath = data_path_binary_search + "p_correct_set.json"
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
    if ifplot:
        plot_general(x=p_correct_set,y_list=mean_ed_list,cdl_style_list=["ro-"],legend_list=["LCPB"],
                 linewidth_list=[3],x_label=u"测量正确的概率",y_label=u"平均编辑距离",
                 pic_path=pic_path)

"""
变化拓扑规模
"""
def plot_topo_zoo():
    pic_path = data_path_binary_search+"topo_var.pdf"
    num_path_list = []
    mean_ed_list = [[]]
    for overlay_nodes,vector_tree in topo_zoo_set:
        filename = data_path_binary_search+str(vector_tree)+".json"
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
    mean_num_list = [[],[],[]]
    for overlay_nodes, vector_tree in topo_zoo_set:
        filename = data_path_binary_search + str(vector_tree) + ".json"
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
        mean_num_list[2].append(num_leaf*(num_leaf-1)/2)
        print(np.mean(num_topo3),num_leaf*(num_leaf-1)*(num_leaf-2)/6,num_leaf*(num_leaf-1)/2)
    pic_path = data_path_binary_search+"measure_complexity.pdf"
    plot_general(x=num_path_list,y_list=mean_num_list,cdl_style_list=["ro-","cs--","b.--"],legend_list=["LCPB","TST","RNJ"],
                 linewidth_list=[3,2,2],x_label=u"路径数量", y_label=u"平均消耗的测量数量",pic_path=pic_path)

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

def plot_RNJ(ifplot=False):
    mean_ed_list = []
    p_correct_set = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
    data_path_rnj = "../data/RNJ/result/"
    for overlay_node_set, tree_vector in topo_zoo_set:
        mean_ed = []
        num_path = len(overlay_node_set)-1
        filename = data_path_rnj+"p_correct_set_"+str(num_path)+".json"
        f = open(filename,"r")
        data_dict = json.load(f)
        p_correct_set = data_dict.get("probability")
        VTree = data_dict.get("tree_vector")
        E = VTreetoE(VTree)
        times = data_dict.get("experiment_times")
        inferred_E_dict = data_dict.get("inferred_E")
        for p_correct in p_correct_set:
            inferred_E_list = inferred_E_dict[p_correct]
            cnt = 0
            sum_ed = 0
            for inferred_E in inferred_E_list:
                ed = calEDbyzss(E, inferred_E)
                if ed == 0:
                    cnt += 1
                sum_ed += ed
            mean_ed.append(sum_ed/len(inferred_E_list))
        print(mean_ed)
        mean_ed_list.append(mean_ed)
    if ifplot:
        plot_general(x=p_correct_set,y_list=mean_ed_list)

def plot_BS():
    mean_ed_list = []
    p_correct_set = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
    for overlay_node_set, tree_vector in topo_zoo_set:
        mean_ed = []
        num_path = len(overlay_node_set)-1
        filename = data_path_binary_search+"p_correct_set_"+str(num_path)+".txt"
        f = open(filename,"r")
        data_dict = json.load(f)
        p_correct_set = data_dict.get("probability")
        VTree = data_dict.get("tree_vector")
        E = VTreetoE(VTree)
        times = data_dict.get("experiment_times")
        inferred_vector_dict = data_dict.get("new_tree_vector")
        for p_correct in p_correct_set:
            inferred_vector_list = inferred_vector_dict[p_correct]
            cnt = 0
            sum_ed = 0
            for inferred_vector in inferred_vector_list:
                E = VTreetoE(VTree)
                inferredE = VTreetoE(inferred_vector)
                ed = calEDbyzss(E, inferredE)
                if ed == 0:
                    cnt += 1
                sum_ed += ed
            mean_ed.append(sum_ed/len(inferred_vector_list))
        print(mean_ed)
        mean_ed_list.append(mean_ed)

def plot_direct_p_correct():
    p_correct_set = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
    num_path_set = [4,5,6,7,8,9]
    LCPB = [
        [0.0, 0.91, 2.19, 2.62, 3.52, 3.23, 3.92, 3.96, 4.08],
        [0.0, 1.58, 2.51, 2.63, 3.6, 4.48, 4.22, 4.89, 5.23],
        [0.0, 2.32, 3.1, 5.12, 5.85, 6.39, 6.93, 7.32, 6.98],
        [0.0, 4.6, 6.65, 8.45, 9.11, 9.39, 9.85, 10.26, 9.63],
        [0.0, 5.12, 8.46, 10.07, 11.12, 11.35, 11.44, 12.5, 12.52],
        [0, 6.81, 10.65, 12.79, 12.8, 14.28, 14.75, 15.12, 15.8],
        # [0.96, 7.09, 11.0, 13.59, 14.61, 15.29, 15.55, 16.34, 16.72]
    ]
    RNJ = [
        [0.0, 0.62, 1.94, 1.71, 2.37, 3.24, 3.1, 3.17, 3.59],
        [0.0, 1.14, 2.04, 2.64, 3.79, 3.74, 4.69, 4.55, 4.58],
        [0.0, 2.07, 3.7, 4.53, 5.65, 5.47, 6.32, 6.08, 6.43],
        [0.0, 2.95, 5.34, 6.61, 7.9, 8.59, 8.78, 8.75, 8.64],
        [0.0, 4.2, 7.82, 8.97, 8.85, 10.26, 10.62, 10.47, 11.48],
        [0.0, 5.92, 9.96, 10.8, 12.05, 13.01, 13.09, 13.77, 13.44],
        # [0.0, 7.31, 10.26, 12.58, 14.24, 14.56, 14.98, 15.09, 15.09]
    ]
    for index,p_correct in enumerate(p_correct_set):
        pic_path = data_path_binary_search + "p_correct_"+str(p_correct)+".pdf"
        y = [[],[]]
        for item in LCPB:
            y[0].append(item[index])
        for item in RNJ:
            y[1].append(item[index])
        plot_general(x=num_path_set,y_list=y,cdl_style_list=["ro-","cs--"],legend_list=["LCBS","RNJ"],
                     linewidth_list=[3,2],x_label=u"路径数量",y_label=u"平均编辑距离",pic_path=pic_path)

def plot_direct_num_path():
    p_correct_set = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
    num_path_set = [4, 5, 6, 7, 8, 9]
    LCPB = [
        [0.0, 0.91, 2.19, 2.62, 3.52, 3.23, 3.92, 3.96, 4.08],
        [0.0, 1.58, 2.51, 2.63, 3.6, 4.48, 4.22, 4.89, 5.23],
        [0.0, 2.32, 3.1, 5.12, 5.85, 6.39, 6.93, 7.32, 6.98],
        [0.0, 4.6, 6.65, 8.45, 9.11, 9.39, 9.85, 10.26, 9.63],
        [0.0, 5.12, 8.46, 10.07, 11.12, 11.35, 11.44, 12.5, 12.52],
        [0, 6.81, 10.65, 12.79, 12.8, 14.28, 14.75, 15.12, 15.8],
        # [0.96, 7.09, 11.0, 13.59, 14.61, 15.29, 15.55, 16.34, 16.72]
    ]
    RNJ = [
        [0.0, 0.62, 1.94, 1.71, 2.37, 3.24, 3.1, 3.17, 3.59],
        [0.0, 1.14, 2.04, 2.64, 3.79, 3.74, 4.69, 4.55, 4.58],
        [0.0, 2.07, 3.7, 4.53, 5.65, 5.47, 6.32, 6.08, 6.43],
        [0.0, 2.95, 5.34, 6.61, 7.9, 8.59, 8.78, 8.75, 8.64],
        [0.0, 4.2, 7.82, 8.97, 8.85, 10.26, 10.62, 10.47, 11.48],
        [0.0, 5.92, 9.96, 10.8, 12.05, 13.01, 13.09, 13.77, 13.44],
        # [0.0, 7.31, 10.26, 12.58, 14.24, 14.56, 14.98, 15.09, 15.09]
    ]
    for index, num_path in enumerate(num_path_set):
        pic_path = data_path_binary_search + "num_path_" + str(num_path) + ".pdf"
        y = [LCPB[index],RNJ[index]]
        plot_general(x=p_correct_set, y_list=y, cdl_style_list=["ro-", "cs--"], legend_list=["LCBS", "RNJ"],
                 linewidth_list=[3, 2], x_label=u"测量正确概率", y_label=u"平均编辑距离", pic_path=pic_path)

if __name__ == '__main__':
    # plot_BS()
    # plot_topo_zoo()
    plot_measure_complexity()
    # plot_RNJ()
    # plot_direct_p_correct()
    # plot_direct_num_path()
    '''
    [0.0, 0.91, 2.19, 2.62, 3.52, 3.23, 3.92, 3.96, 4.08]
    [0.0, 1.58, 2.51, 2.63, 3.6, 4.48, 4.22, 4.89, 5.23]
    [0.0, 2.32, 3.1, 5.12, 5.85, 6.39, 6.93, 7.32, 6.98]
    [0.0, 4.6, 6.65, 8.45, 9.11, 9.39, 9.85, 10.26, 9.63]
    [0.0, 5.12, 8.46, 10.07, 11.12, 11.35, 11.44, 12.5, 12.52]
    [1.12, 6.81, 10.65, 12.79, 12.8, 14.28, 14.75, 15.12, 15.8]
    [0.96, 7.09, 11.0, 13.59, 14.61, 15.29, 15.55, 16.34, 16.72]
    
    [0.0, 0.62, 1.94, 1.71, 2.37, 3.24, 3.1, 3.17, 3.59]
    [0.0, 1.14, 2.04, 2.64, 3.79, 3.74, 4.69, 4.55, 4.58]
    [0.0, 2.07, 3.7, 4.53, 5.65, 5.47, 6.32, 6.08, 6.43]
    [0.0, 2.95, 5.34, 6.61, 7.9, 8.59, 8.78, 8.75, 8.64]
    [0.0, 4.2, 7.82, 8.97, 8.85, 10.26, 10.62, 10.47, 11.48]
    [0.0, 5.92, 9.96, 10.8, 12.05, 13.01, 13.09, 13.77, 13.44]
    [0.0, 7.31, 10.26, 12.58, 14.24, 14.56, 14.98, 15.09, 15.09]
    '''


