#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'Logger'.py
@ide     : 'PyCharm'
@time    : '2020'-'12'-'16' '11':'55':'25'
@contact : zongwang.zhang@outlook.com
'''

import logging
import time

class Logger:
    def __init__(self,logger_name,log_name,log_level=logging.INFO,if_file=True,if_console=False):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        #  输出到文件
        if if_file:
            date = time.asctime()
            now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            log_name += now_time
            fh = logging.FileHandler('log/'+log_name+".log",encoding="utf-8")
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

        # 输出到Console
        if if_console:
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            sh.setLevel(log_level)
            self.logger.addHandler(sh)


    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def war(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def cri(self, message):
        self.logger.critical(message)


