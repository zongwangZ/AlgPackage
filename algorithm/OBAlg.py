#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'OCCAMAlg'.py
@ide     : 'PyCharm'
@time    : '2020'-'11'-'27' '15':'27':'09'
@contact : zongwang.zhang@outlook.com
'''
from pulp import *
import networkx as nx
from util import tool
from matplotlib import pyplot as plt
import logging.config

class OCCAMAlg:
    def __init__(self,network, host_node_set, alpha=0.5):
        self.__init_logger()
        self.__network = network
        self.__alpha = alpha
        self.__H = host_node_set
        self.__init_V()
        self.__init_variables()
        self.__init_objective()
        self.__init_constraints()
        self.G = self.__createNetwork()
        # self.exportModel()


    def getOutcome(self):
        for key in self.__w_ij:
            v = self.__w_ij.get(key)
            if v.value() == 1:
                self.__logger.info("name:"+v.name+"   value:"+str(v.value()))
        m_S_T = []
        for S in self.__H:
            for T in self.__H:
                m_S_T.append(self.__m_S_j.get("{0}^{1}".format(S, T)))
        for v in m_S_T:
            if v.value() != 0:
                self.__logger.info("name:"+v.name+"   value:"+str(v.value()))

        for v in self.__prob.variables():
            self.__logger.debug("name:"+v.name+"   value:"+str(v.value()))

    def solve(self):
        # self.__prob.writeLP("ILP_problem")
        # solver = CPLEX_CMD(keepFiles=True,options=['epgap = 0.25'])
        # solver = PULP_CBC_CMD(gapRel=0.15,timeLimit=1200)
        solver = PULP_CBC_CMD(gapRel=0.15)
        # solver = PULP_CBC_CMD(timeLimit=3600)
        self.__prob.solve(solver)
        self.__logger.info(LpStatus[self.__prob.status])

    def __init_constraints(self):
        self.__init_constraint1()  # Path sharing
        self.__init_constraint2()  # Distance metrics
        self.__init_constraint3()  # Source tree property
        self.__init_constraint4()  # Source-oblivous path
        self.__init_constraint5()  # Populating the d_{i,j}^T variables
        self.__init_constraint6()  # constraints to calculate distance
        self.__init_constraint7()  # Tracing a host-to-host path
        self.__init_constraint8_11()  # Boundary  conditions
        self.__init_constraint12()  # Boundary  conditions
        self.__init_constraint13()  # extra constraint
        self.__init_constraint14()  # extra constraint 2


    def __init_constraint1(self):
        """
        Path sharing，公共路径长度 约束
        PSM(S,T_1,T_2)<PSM(S,T_2,T_3)
        公式含义：
        公共路径(S,T_1,T_2)上的节点多于(S,T_2,T_3)
        其中涉及二元变量相乘，即为and操作，可以转换为一组线性的约束
        xy -> z<=x, z<=y, z>=x+y-1
        :return:
        """
        self.__logger.debug("constraint1 init")
        for S in self.__H:
            for T_1 in self.__H:
                for T_2 in self.__H:
                    for T_3 in self.__H:
                        if self.__is_PSM(S,T_1,T_2,T_3):
                            self.__logger.debug("PSM:"+str((S,T_1,T_2,T_3)))
                            LHS = []
                            for i in self.__V:
                                v_1 = self.__v_i_ST.get("{}".format(i)+"^"+"{" + "{},{}".format(S, T_1) + "}")
                                v_2 = self.__v_i_ST.get("{}".format(i) + "^" + "{" + "{},{}".format(S, T_2) + "}")
                                LHS_i = LpVariable("LHS_{},{},{},{},{}".format(S,T_1,T_2,T_3,i),cat=LpBinary)
                                self.__prob += LHS_i <= v_1
                                self.__prob += LHS_i <= v_2
                                self.__prob += LHS_i >= v_1+v_2-1
                                LHS.append(LHS_i)
                            RHS = []
                            for i in self.__V:
                                v_2 = self.__v_i_ST.get("{}".format(i) + "^" + "{" + "{},{}".format(S, T_2) + "}")
                                v_3 = self.__v_i_ST.get("{}".format(i) + "^" + "{" + "{},{}".format(S, T_3) + "}")
                                RHS_i = LpVariable("RHS_{},{},{},{},{}".format(S,T_1,T_2,T_3,i),cat=LpBinary)
                                self.__prob += RHS_i <= v_2
                                self.__prob += RHS_i <= v_3
                                self.__prob += RHS_i >= v_2 + v_3 - 1
                                RHS.append(RHS_i)
                            self.__prob += lpSum(LHS)+1 <= lpSum(RHS)
        # print(self.__prob.variables())
        # print(self.__prob.constraints)

    def __is_PSM(self,S,T_1,T_2,T_3):
        """
        判断 PSM(S,T_1,T_2)<PSM(S,T_2,T_3) 是否成立
        <i>这里没有限制T不能等于S<i>
        <i>现在增加限制<i>
        :return:
        """
        # 先排除节点相等的情况
        if S == T_1 or S == T_2 or S == T_3 or T_1 == T_2 or T_1 == T_3 or T_2 == T_3:
            return False
        flag = True
        # from ground truth
        assert isinstance(self.__network.G,nx.Graph)
        path1 = nx.shortest_path(self.__network.G,S,T_1)
        path2 = nx.shortest_path(self.__network.G,S,T_2)
        path3 = nx.shortest_path(self.__network.G, S, T_3)
        path12 = set(path1).intersection(set(path2))
        path23 = set(path2).intersection(set(path3))
        if len(path12) < len(path23):
            flag = True
        else:
            flag = False
        return flag

    def __init_constraint2(self):
        """
        Distance metrics 路径长度 约束
        DM(S,T_1)<DM(S,T_2)
        <i>这里没有限制T不能等于S<i>
        <i>增加限制<i>
        :return:
        """
        self.__logger.debug("constraint2 init")
        for S in self.__H:
            for T_1 in self.__H:
                for T_2 in self.__H:
                    if self.__is_DM(S, T_1, T_2):
                        self.__logger.debug("DM:"+str((S, T_1, T_2)))
                        m1 = self.__m_S_j.get("{0}^{1}".format(T_1, S))
                        m2 = self.__m_S_j.get("{0}^{1}".format(T_2, S))
                        self.__prob += m1+1 <= m2
        # print(self.__prob.constraints)

    def __is_DM(self, S, T_1, T_2):
        """
        判断DM(S,T_1)<DM(S,T_2)是否成立
        :param S:
        :param T_1:
        :param T_2:
        :return:
        """
        if S == T_1 or S == T_2 or T_1 == T_2:
            return False
        flag = True
        path1 = nx.shortest_path(self.__network.G,S,T_1)
        path2 = nx.shortest_path(self.__network.G, S, T_2)
        if len(path1) < len(path2):
            flag = True
        else:
            flag = False
        return flag

    def __init_constraint3(self):
        """
        Source tree property
        :return:
        """
        self.__logger.debug("constraint3 init")
        for S in self.__H:
            for j in self.__V:
                self.__prob += \
                    lpSum([self.__s_S_ij.get("{" + "{},{}".format(i, j) + "}"+"^"+"{}".format(S)) for i in self.__V]) <= 1
        # print(self.__prob.constraints)

    def __init_constraint4(self):
        """
        Source-oblivous path
        :return:
        """
        self.__logger.debug("constraint4 init")
        for T in self.__H:
            for i in self.__V:
                self.__prob += \
                    lpSum([self.__d_T_ij.get("{" + "{},{}".format(i, j) + "}" + "^" + "{}".format(T)) for j in
                          self.__V]) <= 1
        # print(self.__prob.constraints)

    def __init_constraint5(self):
        """
        Populating the d_{i,j}^T variables
        :return:
        """
        self.__logger.debug("constraint5 init")
        self.__M = 100
        for T in self.__H:  # 这里的T 是否正确
            for i in self.__V:
                for j in self.__V:
                    d_T_ij = self.__d_T_ij.get("{" + "{},{}".format(i, j) + "}" + "^" + "{}".format(T))
                    # 中间项
                    D_sum = []
                    for S in self.__H:
                        d_s = LpVariable("d_{},{},{},{}".format(S,T,i,j),cat=LpBinary)
                        s_S_ij = self.__s_S_ij.get("{" + "{},{}".format(i, j) + "}"+"^"+"{}".format(S))
                        v_j_ST = self.__v_i_ST.get("{}".format(j)+"^"+"{" + "{},{}".format(S, T) + "}")
                        self.__prob += d_s <= s_S_ij
                        self.__prob += d_s <= v_j_ST
                        self.__prob += d_s >= s_S_ij + v_j_ST - 1
                        D_sum.append(d_s)
                    # 右边不等式
                    self.__prob += lpSum(D_sum) <= self.__M * d_T_ij

                    # 左边不等式
                    self.__prob += lpSum(D_sum) >= -1 * self.__M * (1-d_T_ij) + 1  # 不一定正确
        # print(self.__prob.constraints)

    def __init_constraint6(self):
        """
        constraints to calculate distance
        m_S^S = 0
        :return:
        """
        self.__logger.debug("constraint6 init")
        uper_bound = 10
        for S in self.__H:
            self.__prob += self.__m_S_j.get("{0}^{1}".format(S,S)) == 0
        for j in self.__V:
            for S in self.__H:
                M_sum = []
                S_sum = []
                for i in self.__V:
                    m_i = LpVariable("m_{},{},{}".format(S,i,j),cat=LpInteger)
                    s_S_ij = self.__s_S_ij.get("{" + "{},{}".format(i, j) + "}"+"^"+"{}".format(S))
                    m_S_i = self.__m_S_j.get("{0}^{1}".format(i,S))
                    self.__prob += m_i <= uper_bound * s_S_ij  # z<= Ib
                    self.__prob += m_i <= m_S_i  # z<=i
                    self.__prob += m_i >= m_S_i-(1-s_S_ij)*uper_bound  # z >= i-(1-b)*I
                    self.__prob += m_i >= 0  # z>=0
                    M_sum.append(m_i)
                    S_sum.append(s_S_ij)
                m_S_j = self.__m_S_j.get("{0}^{1}".format(j,S))
                self.__prob += m_S_j == lpSum(M_sum)+lpSum(S_sum)
        # print(self.__prob.constraints)

    def __init_constraint7(self):
        """
        Tracing a host-to-host path
        :return:
        """
        self.__logger.debug("constraint7 init")
        for S in self.__H:
            for T in self.__H:
                for i in list(set(self.__V).difference(set(self.__H))):
                    v_i_ST = self.__v_i_ST.get("{}".format(i)+"^"+"{" + "{},{}".format(S, T) + "}")
                    V_sum = []
                    for j in self.__V:
                        v_j_ST = self.__v_i_ST.get("{}".format(j)+"^"+"{" + "{},{}".format(S, T) + "}")
                        s_S_ij = self.__s_S_ij.get("{" + "{},{}".format(i, j) + "}"+"^"+"{}".format(S))
                        v_i = LpVariable("v_{},{},{},{}".format(S,T,i,j),cat=LpBinary)
                        self.__prob += v_i <= v_j_ST
                        self.__prob += v_i <= s_S_ij
                        self.__prob += v_i >= v_j_ST + s_S_ij-1
                        V_sum.append(v_i)
                    self.__prob += v_i_ST == lpSum(V_sum)
        # print(self.__prob.constraints)

    def __init_constraint8_11(self):
        """
        Boundary  conditions
        对应论文中 10-13
        :return:
        """
        self.__logger.debug("constraint8-11 init")
        for S in self.__H:
            # constraint10
            self.__prob += lpSum([self.__s_S_ij.get("{" + "{},{}".format(S, j) + "}"+"^"+"{}".format(S)) for j in self.__V]) == 1

            # constraint11
            for T in self.__H:
                if T != S:
                    self.__prob += lpSum(
                        [self.__s_S_ij.get("{" + "{},{}".format(j, T) + "}" + "^" + "{}".format(S)) for j in self.__V]) == 1

            # constraint12
            self.__prob += lpSum(
                [self.__s_S_ij.get("{" + "{},{}".format(j, S) + "}" + "^" + "{}".format(S)) for j in self.__V]) == 0

            # constraint13
            for T in self.__H:
                if T != S:
                    self.__prob += lpSum(
                        [self.__s_S_ij.get("{" + "{},{}".format(T, j) + "}" + "^" + "{}".format(S)) for j in self.__V]) == 0

        # print(self.__prob.constraints)

    def __init_constraint12(self):
        """
        Boundary  conditions
        对应文章中的14
        :return:
        """
        self.__logger.debug("constraint12 init")
        for S in self.__H:  # 不是很确定
            for k in self.__V:  # 不是很确定
                for j in list(set(self.__V).difference({S})):
                    s_S_jk = self.__s_S_ij.get("{" + "{},{}".format(j, k) + "}"+"^"+"{}".format(S))
                    self.__prob += s_S_jk <= lpSum([self.__s_S_ij.get("{" + "{},{}".format(i, j) + "}"+"^"+"{}".format(S)) for i in self.__V])
        # print(self.__prob.constraints)

    def __init_constraint13(self):
        """
        补充对w_{i,j}的约束
        :return:
        """
        self.__logger.debug("constraint13 init")
        for i in self.__V:
            for j in self.__V:
                w_ij = self.__w_ij.get("{" + "{},{}".format(i, j) + "}")

                # OR_S s_{i,j}^S
                S_ij = LpVariable("S_{},{}".format(i,j),cat=LpBinary)
                S_sum = []
                for S in self.__H:
                    s_S_ij = self.__s_S_ij.get("{" + "{},{}".format(i, j) + "}"+"^"+"{}".format(S))
                    S_sum.append(s_S_ij)
                    self.__prob += S_ij >= s_S_ij
                self.__prob += S_ij <= lpSum(S_sum)
                self.__prob += w_ij == S_ij

                # OR_T d_{i,j}^S
                # D_ij = LpVariable("D_{},{}".format(i,j),cat=LpBinary)
                # D_sum = []
                # for T in self.__H:
                #     m_T_ij = self.__d_T_ij.get("{" + "{},{}".format(i, j) + "}" + "^" + "{}".format(T))
                #     D_sum.append(m_T_ij)
                #     self.__prob += D_ij >= m_T_ij
                # self.__prob += D_ij <= lpSum(D_sum)
                # self.__prob += w_ij == D_ij

                # for S in self.__H:
                #     for T in self.__H:
                #         v_i_ST = self.__v_i_ST.get()
                #         v_j_ST = self.__v_i_ST.get()
                #         m_S_i = self.__m_S_j.get()
                #         m_S_j = self.__m_S_j.get()

        # print(self.__prob.constraints)

    def __init_constraint14(self):
        """

        对v_i^{S,T}做限制
        :return:
        """
        self.__logger.debug("constraint14 init")
        for S in self.__H:
            for T in self.__H:
                for i in self.__H:
                    if S != T:
                        if i == S or i == T:
                            self.__prob += self.__v_i_ST.get("{}".format(i)+"^"+"{" + "{},{}".format(S, T) + "}") == 1
                        else:
                            self.__prob += self.__v_i_ST.get("{}".format(i)+"^"+"{" + "{},{}".format(S, T) + "}") == 0
                    elif S == T:
                        self.__prob += self.__v_i_ST.get("{}".format(i)+"^"+"{" + "{},{}".format(S, T) + "}") == 0


    def __init_V(self):
        """
        初始化V，默认为端节点数量的两倍，且从0到N
        :return:
        """
        self.__host_num = len(self.__H)
        self.__node_num = 2*(self.__host_num-1)
        # self.__node_num = 10
        self.__V = []
        for node in range(0, self.__node_num):
            self.__V.append(node)

    def __init_variables(self):
        self.__m_S_j = self.__create_variable1()  # m_j^S
        self.__w_ij = self.__create_variable2()   # w_ij
        self.__v_i_ST = self.__create_variable3()  # v_i^ST
        self.__s_S_ij = self.__create_variable4()  # s_{i,j}_S
        self.__d_T_ij = self.__create_variable5()  # d_{i,j}_T

    def __create_variable1(self):
        """
        m_T^S
        :return:
        """
        upBound = 10
        variables_list = []
        for S in self.__H:
            for j in self.__V:
                variables_list.append("{0}^{1}".format(j,S))
        variables = LpVariable.dict("m",variables_list,lowBound=0, upBound=upBound, cat=LpInteger) # 自行定义了上届
        # print(variables)
        return variables

    def __create_variable2(self):
        """
        w_ij
        :return:
        """
        variables_list = []
        for i in self.__V:
            for j in self.__V:
                variables_list.append("{" + "{},{}".format(i, j) + "}")
        variables = LpVariable.dict("w",variables_list,cat=LpBinary)
        # print(variables)
        return variables

    def __create_variable3(self):
        """
        v_i^ST
        :return:
        """
        variables_list = []
        for i in self.__V:
            for S in self.__H:
                for T in self.__H:
                    variables_list.append("{}".format(i)+"^"+"{" + "{},{}".format(S, T) + "}")
        variables = LpVariable.dict("v",variables_list,cat=LpBinary)
        # print(variables)
        return variables

    def __create_variable4(self):
        """
        s_{i,j}_S
        :return:
        """
        variables_list = []
        for i in self.__V:
            for j in self.__V:
                for S in self.__H:
                    variables_list.append("{" + "{},{}".format(i, j) + "}"+"^"+"{}".format(S))
        variables = LpVariable.dict("s",variables_list,cat=LpBinary)
        # print(variables)
        return variables

    def __create_variable5(self):
        """
        d_{i,j}_T
        :return:
        """
        variables_list = []
        for i in self.__V:
            for j in self.__V:
                for T in self.__H:
                    variables_list.append("{" + "{},{}".format(i, j) + "}" + "^" + "{}".format(T))
        variables = LpVariable.dict("d", variables_list, cat=LpBinary)
        # print(variables)
        return variables

    def __init_objective(self):
        """
        目标函数 同时优化 最少的链路数量和最短路径长度
        :return:
        """
        self.__prob = LpProblem("OCCAM",LpMinimize)
        m_S_T = []
        for S in self.__H:
            for T in self.__H:
                m_S_T.append(self.__m_S_j.get("{0}^{1}".format(S,T)))
        component1 = self.__alpha*lpSum(m_S_T)
        component2 = (1-self.__alpha)*lpSum(self.__w_ij)  # 最少链路
        self.__prob += component1 + component2
        # print(self.__prob.objective)

    def inference(self):
        """
        对应文章算法1 GRAPH-CONSTRUCT-1
        未检验
        :return:
        """
        print("do_inference")
        pi = {}
        for S in self.__H:
            for T in self.__H:
                if S != T:
                    key = "{},{}".format(S, T)
                    if key not in pi:
                        pi[key] = []
                    i_0 = T
                    pi[key].append(i_0)
                    i_k_1 = i_0
                    i_k_2 = i_0
                    while i_k_2 != S:
                        for i_k_2 in self.__V:
                            if self.__s_S_ij.get("{" + "{},{}".format(i_k_2, i_k_1) + "}"+"^"+"{}".format(S)).value() == 1:
                                pi[key].append(i_k_2)
                                self.G.add_edge(i_k_1,i_k_2)
                                i_k_1 = i_k_2
                                break
        for key in pi:
            pi[key].reverse()
        self.__logger.info(pi)

    def plot_inferred_graph(self):
        """
        将推断的拓扑绘制
        :return:
        """
        nx.draw(self.G, with_lables=True)
        plt.show()

    def __createNetwork(self):
        G = nx.Graph()
        return G

    def true_objective_value(self):
        """
        输出真实目标函数的值
        最短路径长度+最少数量边
        :return:
        """
        assert isinstance(self.__network.G,nx.Graph)
        sum_component1 = 0
        for S in self.__H:
            for T in self.__H:
                path = nx.shortest_path(self.__network.G,S,T)
                sum_component1 += len(path)-1
        sum_component2 = len(self.__network.G.edges)*2
        value = self.__alpha * sum_component1 + (1-self.__alpha) * sum_component2
        self.__logger.info("true objective value:"+str(value))
        return value

    def __init_logger(self):
        """
        初始化logger
        :return:
        """
        # self.logger = Logger(logger_name=self.__class__.__name__, log_name=self.__class__.__name__)
        # self.logger.info("初始化"+self.__class__.__name__)
        logging.config.fileConfig('util/logging.conf')
        self.__logger = logging.getLogger('applog')

    def exportModel(self):
        """
        将模型导出为dict
        :return:
        """
        model_data = self.__prob.to_dict()
        self.__logger.info(str(model_data))