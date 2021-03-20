#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'rnj_m2_generator'.py
@ide     : 'PyCharm'
@time    : '2021'-'03'-'13' '20':'38':'05'
@contact : zongwang.zhang@outlook.com
'''
from network import Network
from data_generator.binary_search_generator import M2Measure
import logging
import numpy as np
from util.tool import EtoVTree, VTreetoE
import json
from data.topo_zoo.init import topo_zoo_set
data_path = "../data/RNJ/"
def gen_data_rnj_p_correct():
    p_correct_set = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
    times = 100
    logger = logging.getLogger("data_generator")
    for overlay_node_set, tree_vector in topo_zoo_set:
        data_dict = {}
        E = VTreetoE(tree_vector)
        num_path = len(overlay_node_set) - 1
        data_dict["probability"] = [str(p) for p in p_correct_set]
        data_dict["tree_vector"] = tree_vector
        data_dict["experiment_times"] = times
        data_dict["num_leaf"] = num_path
        network = Network(overlay_node_set, E, logger=logger)
        data_dict["topo_2_data"] = {}
        for p_correct in p_correct_set:
            data_dict["topo_2_data"][str(p_correct)] = []
            for i in range(times):
                m2Measure = M2Measure(network=network, p_correct=p_correct)
                ind_m2 = m2Measure.indm2
                measured_m2 = m2Measure.measure_m2_all()
                similarity_matrix = np.zeros((num_path,num_path),dtype=int)
                for i in range(num_path):
                    for j in range(num_path):
                        if i<j:
                            index = ind_m2[i][j]-1
                            similarity = measured_m2[index]
                            similarity_matrix[i][j] = similarity_matrix[j][i] = similarity
                # print(measured_m2)
                # print(similarity_matrix)
                similarity_matrix = similarity_matrix.tolist()
                data_dict["topo_2_data"][str(p_correct)].append(similarity_matrix)
        filename = data_path+"p_correct_set_"+str(num_path)+".txt"
        with open(filename,"w") as f:
            json.dump(data_dict,f,indent=4)
if __name__ == '__main__':
    gen_data_rnj_p_correct()

