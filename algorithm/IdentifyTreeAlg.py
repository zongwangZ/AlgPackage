#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'IdentifyTreeAlg'.py
@ide     : 'PyCharm'
@time    : '2020'-'11'-'30' '10':'04':'29'
@contact : zongwang.zhang@outlook.com
'''
from util.EditDistanceUtil import *
import logging.config
class IdentifyTreeAlg:
    """
   Topoloy Discovery Using Path Interference中Algorithm2方法的实现
    """
    def __init__(self,network,overlay_node_set:list,interference_matrix:np.ndarray):
        self.__init_logger()
        self.__network = network
        self.__overlay_node_set = overlay_node_set
        self.__overlay_node_num = len(overlay_node_set)
        self.__interference_matrix = interference_matrix
        self.__init_tunnels()
        self.G = self.__createNetwork()
        self.__node_no = self.__overlay_node_num+1


    def __init_tunnels(self):
        self.__tunnel_list = []
        for node1 in self.__overlay_node_set:
            for node2 in self.__overlay_node_set:
                if node1 != node2:
                    self.__tunnel_list.append([node1, node2])
        self.__tunnel_num = self.__overlay_node_num*(self.__overlay_node_num-1)

    def __createNetwork(self):
        G = nx.Graph()
        return G

    def inference(self):
        O = self.__overlay_node_set.copy()
        # 对应步骤1
        for overlay_node in O:
            self.G.add_node(overlay_node)
        interference_matrix = self.__interference_matrix.copy()
        tunnel_list = self.__tunnel_list.copy()
        self.__recovering_iterative(O,interference_matrix,tunnel_list)

    def __recovering_iterative(self,O,interference_matrix,tunnel_list):
        # 对应步骤2，3
        if len(O) == 1:
            return
        if len(O) == 2:
            self.G.add_edge(O[0],O[1])
            return
        #对应步骤4
        k_star = np.argmax(np.sum(interference_matrix,0))
        k_start1 = tunnel_list[k_star+1][0]
        X_k_star = [k_start1]
        # 对应步骤5
        for overlay_node in O:
            if overlay_node != k_start1:
                are_sibling = self.__are_sibling(interference_matrix,tunnel_list,k_start1,overlay_node)
                if are_sibling == True:
                    X_k_star.append(overlay_node)
        # 对应步骤6
        p_X_k_star = self.__node_no
        self.__node_no += 1
        for node in X_k_star:
            self.G.add_edge(node,p_X_k_star)
        nx.draw(self.G,with_labels=True)
        plt.show()
        # 对应步骤7
        for node in X_k_star:
            delete_index = []
            if node != k_start1:
                O.remove(node)
                for tunnel in tunnel_list:
                    if tunnel[0] == node or tunnel[1] == node:
                        index = tunnel_list.index(tunnel)
                        delete_index.append(index)
                interference_matrix = np.delete(interference_matrix,delete_index,axis=0)
                interference_matrix = np.delete(interference_matrix, delete_index, axis=1)
                delete_tunnels = []
                for tunnel_index in delete_index:
                    delete_tunnels.append(tunnel_list[tunnel_index])
                for delete_tunnel in delete_tunnels:
                    tunnel_list.remove(delete_tunnel)
        # 对应步骤8
        O[O.index(k_start1)] = p_X_k_star
        for tunnel in tunnel_list:
            if tunnel[0] == k_start1:
                tunnel[0] = p_X_k_star
            elif tunnel[1] == k_start1:
                tunnel[1] = p_X_k_star

        self.__recovering_iterative(O,interference_matrix,tunnel_list)



    def __are_sibling(self,interference_matrix,tunnel_list,i,j):
        """
        判断i和j节点是否共有同一个父节点
        :param i:
        :param j:
        :return:
        """
        # 对应步骤1
        k_tunnel = [i,j]
        k = tunnel_list.index(k_tunnel)+1
        # 对应步骤2
        for l in range(1,len(tunnel_list)+1):
            if l != k:
                if interference_matrix[k-1][l-1] == 1:
                    l_tunnel = tunnel_list[l-1]
                    l_1 = l_tunnel[0]
                    l_l = l_tunnel[1]
                    if l_1 != i and l_l != j:
                        return False
        k_tunnel = [j,i]
        k = tunnel_list.index(k_tunnel) + 1
        for l in range(1,len(tunnel_list)+1):
            if l != k:
                if interference_matrix[k - 1][l - 1] == 1:
                    l_tunnel = tunnel_list[l - 1]
                    l_1 = l_tunnel[0]
                    l_l = l_tunnel[1]
                    if l_1 != j and l_l != i:
                        return False
        return True

    def compute_ed(self):
        nx.draw(self.__network.G,with_labels=True)
        plt.show()
        nx.draw(self.G,with_labels=True)
        plt.show()
        root = self.__overlay_node_set[0]
        R = self.__overlay_node_set[1:]
        EditDistanceUtil().compute(self.__network.G,self.G,R,root=0)

    def __init_logger(self):
        """
        初始化logger
        :return:
        """
        # self.logger = Logger(logger_name=self.__class__.__name__, log_name=self.__class__.__name__)
        # self.logger.info("初始化"+self.__class__.__name__)
        logging.config.fileConfig('log_config/logging.conf')
        self.__logger = logging.getLogger('applog')
