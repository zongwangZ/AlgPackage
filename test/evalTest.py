#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'evalTest'.py
@ide     : 'PyCharm'
@time    : '2020'-'12'-'28' '20':'35':'56'
@contact : zongwang.zhang@outlook.com
'''
eval('"log"+__import__("time").strftime("%Y-%m-%d_%H-%M-%S", __import__("time").localtime())+".txt"')
eval('"log"+str(__import__("datetime").datetime.now().day)+"_"+str(__import__("datetime").datetime.now().hour)+"_"+str(__import__("datetime").datetime.now().minute)+".txt"')