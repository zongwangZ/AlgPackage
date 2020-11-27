#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'ILPAlg'.py
@ide     : 'PyCharm'
@time    : '2020'-'11'-'13' '20':'27':'52'
@contact : zongwang.zhang@outlook.com
'''
from pulp import *
import numpy as np


class ILPAlg:
    def __init__(self, node_num, overlay_node_num, interference_matrix):
        self.__node_num = node_num
        self.__overlay_node_num = overlay_node_num
        # self.__interference_matrix = interference_matrix
        # self.__interference_matrix = np.array(
        #     [[0, 1, 0, 0, 0, 1],
        #      [1, 0, 0, 1, 0, 1],
        #      [0, 0, 0, 1, 1, 0],
        #      [0, 1, 1, 0, 0, 0],
        #      [0, 0, 1, 0, 0, 1],
        #      [1, 0, 0, 0, 1, 0]], dtype=int)
        self.__interference_matrix = np.array(
            [[1, 1, 0, 0, 0, 1],
             [1, 1, 0, 1, 0, 1],
             [0, 0, 1, 1, 1, 0],
             [0, 1, 1, 1, 0, 0],
             [0, 0, 1, 0, 1, 1],
             [1, 0, 0, 0, 1, 1]], dtype=int)
        # self.__interference_matrix = np.array(
        #     [[1, 0],
        #      [0, 1],], dtype=int)

        self.__init_tunnels()
        self.__init_objective()
        self.__init_constrains()

    def getOutcome(self):
        # for v in self.__prob.variables():
        #     print(v.name, "=", v.varValue)
        for v in self.__prob.variables():
            for key in self.__x_ij:
                if self.__x_ij[key].name == v.name:
                    print(v.name, "=", v.varValue)

    def solve(self):
        self.__prob.writeLP("ILP_problem")
        self.__prob.solve()

    def __init_tunnels(self):
        self.__tunnel_num = self.__overlay_node_num * (self.__overlay_node_num - 1)
        self.__tunnel_list = []
        for i in range(1, self.__overlay_node_num + 1):
            for j in range(1, self.__overlay_node_num + 1):
                if i != j:
                    self.__tunnel_list.append((i, j))

    def __init_objective(self):
        """
        初始化目标函数
        :return:
        """
        self.__prob = LpProblem("ILP_inference", LpMinimize)
        variables_list = []
        for i in range(1, self.__node_num + 1):
            for j in range(1, self.__node_num + 1):
                variables_list.append("{" + "{},{}".format(i, j) + "}")
        self.__x_ij = LpVariable.dict("x", variables_list, cat=LpBinary)  # x_{i,j}变量
        self.__prob += lpSum(self.__x_ij)

    def __init_constrains(self):
        self.__init_constrain1()
        self.__init_constrain2()
        self.__init_constrain3()
        self.__init_constrain4()
        self.__init_constrain5()
        self.__init_constrain6()

    def __init_constrain1(self):
        """
        x_ij的定义
        :return:
        """
        variables_list = []
        for i in range(1, self.__node_num + 1):
            for j in range(1, self.__node_num + 1):
                for k in range(1, self.__tunnel_num + 1):
                    variables_list.append("{" + "{},{}".format(i, j) + "}" + "^{}".format(k))
        self.__x_ij_l = LpVariable.dict("x", variables_list, cat=LpBinary)  # x_{i,j}^l变量

        # x_{i,i}^l = 0
        for i in range(1, self.__node_num + 1):
            for j in range(1, self.__node_num + 1):
                if i == j:
                    for k in range(1, self.__tunnel_num + 1):
                        self.__prob += self.__x_ij_l.get("{" + "{},{}".format(i, j) + "}" + "^{}".format(k)) == 0

        # x_{i,j}的定义
        for i in range(1, self.__node_num + 1):
            for j in range(1, self.__node_num + 1):
                x_ij = self.__x_ij.get("{" + "{0},{1}".format(i, j) + "}")
                sum_x_ij_l = 0
                for k in range(1, self.__tunnel_num + 1):
                    x_ij_l = self.__x_ij_l.get("{" + "{},{}".format(i, j) + "}" + "^{}".format(k))
                    x_ji_l = self.__x_ij_l.get("{" + "{},{}".format(j, i) + "}" + "^{}".format(k))
                    self.__prob += x_ij >= x_ij_l
                    self.__prob += x_ij >= x_ji_l
                    sum_x_ij_l = sum_x_ij_l + x_ij_l + x_ji_l
                self.__prob += x_ij <= sum_x_ij_l

    def __init_constrain2(self):
        """
        overlay node只与一个underlay node相连
        :return:
        """
        for i in range(1, self.__overlay_node_num + 1):
            variables_list = []
            for j in range(1, self.__node_num + 1):
                variables_list.append(self.__x_ij.get("{" + "{},{}".format(i, j) + "}"))
            self.__prob += lpSum(variables_list) == 1



    def __init_constrain3(self):
        """
        流守恒约数，即流只能开始于overlay node且结束于overlay node
        :return:
        """
        for l in range(1, self.__tunnel_num + 1):
            for j in range(1, self.__node_num + 1):  # self.__node_num self.__overlay_node_num
                tunnel = self.__tunnel_list[l - 1]
                s_lj = 1 if tunnel[0] == j else 0
                d_lj = 1 if tunnel[1] == j else 0
                sum_x_ij_l = []
                sum_x_ji_l = []
                for i in range(1, self.__node_num+1):
                    sum_x_ij_l.append(self.__x_ij_l.get("{" + "{},{}".format(i, j) + "}" + "^{}".format(l)))
                    sum_x_ji_l.append(self.__x_ij_l.get("{" + "{},{}".format(j, i) + "}" + "^{}".format(l)))
                self.__prob += lpSum(sum_x_ij_l) + s_lj == lpSum(sum_x_ji_l) + d_lj

    def __init_constrain4(self):
        """
        解决约束3中存在环的情况
        :return:
        """
        # 创建 u_i^l变量 并增加约束
        u_variable_list = []
        for i in range(1, self.__tunnel_num + 1):
            for j in range(1, self.__node_num + 1):
                u_variable_list.append("{}^{}".format(j, i))
        u_variables = LpVariable.dict("u", u_variable_list, lowBound=0, cat=LpInteger)
        for i in range(1, self.__tunnel_num + 1):
            for j in range(1, self.__node_num + 1):
                self.__prob += u_variables.get("{}^{}".format(j, i)) >= 0

        # 添加约束 防止环产生
        for l in range(1, self.__tunnel_num + 1):
            for i in range(1, self.__node_num + 1):
                for j in range(1, self.__node_num + 1):
                    if i != j:
                        u_i_l = u_variables.get("{}^{}".format(i, l))
                        u_j_l = u_variables.get("{}^{}".format(j, l))
                        x_ij_l = self.__x_ij_l.get("{" + "{},{}".format(i, j) + "}" + "^{}".format(l))

                        self.__prob += u_i_l - u_j_l + self.__node_num * x_ij_l <= self.__node_num - 1

    def __init_constrain5(self):
        """
        interference constraints.
        ** 这里可能需要加上k!=l这个条件 **
        :return:
        """
        r_indexs, c_indexs = np.where(self.__interference_matrix == 0)
        assert isinstance(r_indexs, np.ndarray)
        length = r_indexs.size
        for index in range(length):
            k = r_indexs[index] + 1
            l = c_indexs[index] + 1
            # if k!=l:
            for i in range(1, self.__node_num + 1):
                for j in range(1, self.__node_num + 1):
                    x_ij_k = self.__x_ij_l.get("{" + "{},{}".format(i, j) + "}" + "^{}".format(k))
                    x_ij_l = self.__x_ij_l.get("{" + "{},{}".format(i, j) + "}" + "^{}".format(l))
                    self.__prob += x_ij_k + x_ij_l <= 1

    def __init_constrain6(self):
        """
        interference constraints
        :return:
        """
        r_indexs, c_indexs = np.where(self.__interference_matrix == 1)
        assert isinstance(r_indexs, np.ndarray)
        length = r_indexs.size
        for index in range(length):
            k = r_indexs[index] + 1
            l = c_indexs[index] + 1
            sum_and = []
            if k>l:
                for i in range(1, self.__node_num + 1):
                    for j in range(1, self.__node_num + 1):
                        x_ij_k = self.__x_ij_l.get("{" + "{},{}".format(i, j) + "}" + "^{}".format(k))
                        x_ij_l = self.__x_ij_l.get("{" + "{},{}".format(i, j) + "}" + "^{}".format(l))
                        x_and_ijkl = LpVariable("x_and_{}{}{}{}".format(i,j,k,l),cat=LpBinary)
                        self.__prob += x_and_ijkl <= x_ij_k
                        self.__prob += x_and_ijkl <= x_ij_l
                        self.__prob += x_and_ijkl >= x_ij_k+x_ij_l-1
                        sum_and .append(x_and_ijkl)

                self.__prob += lpSum(sum_and) >= 1
