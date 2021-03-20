#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'runRNJ'.py
@ide     : 'PyCharm'
@time    : '2021'-'03'-'13' '20':'34':'50'
@contact : zongwang.zhang@outlook.com
'''
from algorithm.RNJAlg import RNJAlg
from network import Network
from util.tool import VTreetoE, getLeafNodes,calEDbyzss
import json
data_path = "data/RNJ/"
from data.topo_zoo.init import topo_zoo_set
import logging
def run_sim_p_correct():
    logger = logging.getLogger("simulator")
    for overlay_node_set,tree_vector in topo_zoo_set:
        num_path = len(overlay_node_set)-1
        file_path = data_path+"p_correct_set_"+str(num_path)+".txt"
        file = open(file_path,"r")
        data_dict = json.load(file)
        assert tree_vector == data_dict.get("tree_vector")
        E = VTreetoE(tree_vector)
        times = data_dict.get("experiment_times")
        p_correct_set = data_dict.get("probability")
        data_dict["inferred_E"] = {}
        for p in p_correct_set:
            data_dict["inferred_E"][p] = []
            cnt = 0
            sum_ed = 0
            for i in range(times):
                similarity_matrix = data_dict.get("topo_2_data").get(p)[i]
                network = Network(overlay_node_set,E,logger)
                alg = RNJAlg(network,logger)
                inferred_E = alg.inference(similarity_matrix,threshold=0.5)
                data_dict["inferred_E"][p].append(inferred_E)
                ed = calEDbyzss(E, inferred_E)
                if ed == 0:
                    cnt += 1
                sum_ed += ed
            print("正确率：", cnt / times)
            print("平均编辑距离：", sum_ed / times)
        filename = "data/RNJ/result/p_correct_set_"+str(num_path)+".json"
        with open(filename, "w") as f:
            json.dump(data_dict, f, indent=4)

if __name__ == '__main__':
    run_sim_p_correct()