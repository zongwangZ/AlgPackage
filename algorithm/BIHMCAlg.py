#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'BIHMCAlg'.py
@ide     : 'PyCharm'
@time    : '2021'-'01'-'05' '10':'41':'08'
@contact : zongwang.zhang@outlook.com
'''
"""
贝叶斯后验估计+汉密尔顿蒙特卡洛采样
"""
# import pymc3 as pm
import arviz as az
import numpy as np
import math
import pystan
import pickle
from os.path import exists as file_exists
from logging import config
import logging
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
class BIHMC:

    def __init__(self, sim_data: dict, logger, debug=0,iteration_steps=2000,way="pystan"):
        self.way = way
        self.debug = debug
        self.iteration_steps = iteration_steps
        self.now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # 配置logger
        self.__init_logger(logger)
        # if way == "pystan":
        #     self.config_pystan_logger()


        # 参数
        self.N = sim_data.get("N")
        self.T = sim_data.get("T")
        self.prc = sim_data.get("prc")
        self.ind_m2 = sim_data.get("ind_m2")
        self.ind_m3 = sim_data.get("ind_m3")
        self.true_m2 = sim_data.get("true_m2")
        self.m3_observed = sim_data.get("m3_observed")
        self.r_ns_true = sim_data.get("r_ns_true")
        # 文件路径
        self.init_data_path()

    def init_data_path(self):
        if self.way == "pystan":
            self.data_path_pystan = "data/pystan/"  # pystan 数据的根目录
            os.mkdir(self.data_path_pystan+self.now_time)
            self.data_path_pystan_now = self.data_path_pystan+self.now_time+"/"
            # self.model_path_stan = 'algorithm/m3.stan'  # stan 模型文件
            self.model_path_stan = 'algorithm/new_m3.stan'  # new stan 模型文件
            self.cmodel_path_stan = self.data_path_pystan + "model.pkl"  # 编译后的模型文件
            self.fit_path_stan = self.data_path_pystan_now + "fit.pkl"  # fit路径
            self.fit_samples_path_stan = self.data_path_pystan_now + "fit_samples.csv"  # fit samples的路径
            self.model_data_path_stan = self.data_path_pystan_now + "model_data.txt"  # 模型数据的路径

    def inference(self):
        self.__logger.info("-"*10 + f"using {self.way} build the model" + "-"*10)
        if self.way == "pystan":
            # 加载 stan 模型
            if file_exists(self.cmodel_path_stan):
                sm = self.load_model_pystan(self.cmodel_path_stan)
            else:
                sm = self.gen_model_with_pystan()
                self.save_mode_pystan(sm, self.cmodel_path_stan)
            # 产生模型数据
            model_data = self.gen_model_data_pystan()

            # 进行采样
            fit = self.sampling_pystan(model_data,sm)
            self.save_fit_pystan(sm, fit, self.fit_path_stan)

    def get_outcome(self, way="pystan",ifplot=False):
        if way == "pystan":
            return self.get_outcome_pystan(ifplot)





    def get_outcome_pystan(self,ifplot=False):
        if file_exists(self.fit_path_stan):
            fit = self.load_fit_pystan(self.fit_path_stan)
            la = fit.extract(permuted=True)
            m2_samples = la["m2"]
            # plt.acorr(m2_samples[:,0]) #画什么东西来着
            # plt.show()
            assert isinstance(m2_samples,np.ndarray)
            # self.__logger.info("m2_samples:"+str(m2_samples))
            inferred_m2_real = np.mean(m2_samples,axis=0)
            self.__logger.info("inferred_m2_real:"+str(inferred_m2_real))
            self.__logger.info("self.true_m2:"+str(self.true_m2))

            if ifplot:
                fit.plot()
                plt.show()
                az.plot_trace(fit)
            return inferred_m2_real,self.true_m2


        else:
            print("no fit file generated")
            return

    def gen_model_data_pystan(self):
        model_data = dict(
            do_debug=self.debug,  # 0为不打印，1为打印结果
            T=self.T,
            N=self.N,
            m3_observed=self.m3_observed,
            prc=self.prc,
            index_m2=self.ind_m2,
            index_m3=self.ind_m3,
            net_status = np.random.uniform(0,1,self.T), #对应new_m3.stan，减小难度
            r_ns = self.r_ns_true  # 对应new_m3.stan 减小难度
        )
        # self.__logger.info("model data is "+str(model_data))
        # 保存模型数据
        pystan.misc.stan_rdump(model_data, self.model_data_path_stan)
        self.__logger.debug("保存模型数据到:"+self.model_data_path_stan)
        return model_data

    def sampling_pystan(self,model_data, sm):
        self.__logger.info("-"*10+"pystan sampling"+"-"*10)
        # 马尔可夫链初始化参数,和老师的不太一样
        true_m2 = np.copy(self.true_m2)
        r_ns_true = self.r_ns_true
        N = self.N

        # def model_init():
        #     return dict(m2=true_m2 + np.random.uniform(low=0.0, high=0.5, size=true_m2.size),
        #                 net_status=np.random.uniform(low=0,high=1,size=self.T),
        #                 r_n2 = self.r_ns_true)  # 暂时不进行初始化

        # def model_init():
        #     return dict(m2=true_m2 + np.random.uniform(-0.5,0.5,len(true_m2)),
        #                 )  # 暂时不进行初始化

        def model_init():
            init_m2 = []
            for item in true_m2:
                if item == N-1:
                    init_m2.append(item + np.random.uniform(-1,0.5))
                elif item == 1:
                    init_m2.append(item + np.random.uniform(-0.5,1))
                else:
                    init_m2.append(item + np.random.uniform(-1,1))
            return dict(m2=init_m2,
                        )

        fit = sm.sampling(data=model_data,
                          chains=4,
                          # warmup=1000,
                          init=model_init,
                          iter=self.iteration_steps,
                          seed=20200829,
                          # algorithm="HMC",
                          # control=dict(max_treedepth=12, adapt_delta=0.90),
                          control=dict(max_treedepth=12, adapt_delta=0.9),
                          # control=dict(max_treedepth=15, adapt_delta=0.99),
                          # control=dict(adapt_delta=0.90)
                          )
        self.save_samples_tocsv(fit)
        return fit

    def save_samples_tocsv(self,fit):
        self.__logger.debug("保存samples到:"+self.fit_samples_path_stan)
        df = pystan.misc.to_dataframe(fit)
        df.to_csv(self.fit_samples_path_stan, index=False)


    def save_fit_pystan(self,sm, fit, fit_path):
        """
        参考 https://pystan.readthedocs.io/en/latest/unpickling_fit_without_model.html
        pickle fit的时候，需要同时pickle stan_model
        :param sm:
        :param fit:
        :param fit_path:
        :return:
        """
        fit_model = dict(
            stan_model=sm,
            fit=fit
        )
        with open(fit_path, "wb") as f:
            pickle.dump(fit_model, f)


    def load_fit_pystan(self, fit_path):
        fit = pickle.load(open(fit_path, "rb"))["fit"]
        return fit

    def gen_model_with_pystan(self):
        sm = pystan.StanModel(file=self.model_path_stan)
        self.__logger.info("生成编译的stan model:" + sm.model_name)
        return sm

    def load_model_pystan(self, cmodel_path):
        sm = pickle.load(open(cmodel_path, 'rb'))
        assert isinstance(sm,pystan.StanModel)
        self.__logger.debug("pystan model 加载成功  " + sm.model_name)
        return sm

    def save_mode_pystan(self, sm, cmodel_path_stan):
        with open(cmodel_path_stan, 'wb') as f:
            pickle.dump(sm, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.__logger.debug("保存stan model 到"+cmodel_path_stan)

    def __init_logger(self, logger):
        """
        初始化logger
        :return:
        """
        if logger is not None:
            self.__logger = logger
        else:
            logging.config.fileConfig('util/logging.conf')
            self.__logger = logging.getLogger('applog')

    def config_pystan_logger(self):
        logger = logging.getLogger("pystan")
        logger_path = "log/pystan" + time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()) + ".log"
        fh = logging.FileHandler(logger_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        # optional step
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)


