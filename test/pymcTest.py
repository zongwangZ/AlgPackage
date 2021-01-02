#!/usr/bin/env python
# encoding: utf-8
'''
@project : 'AlgPackage'
@author  : '张宗旺'
@file    : 'pymcTest'.py
@ide     : 'PyCharm'
@time    : '2020'-'12'-'29' '13':'47':'56'
@contact : zongwang.zhang@outlook.com
'''
import pymc3 as pm
import numpy.random as npr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import Counter
import seaborn as sns

sns.set_style('white')
sns.set_context('poster')

# import warnings
# warnings.filterwarnings('ignore')

from random import shuffle
total = 30
n_heads = 15
n_tails = total - n_heads
tosses = [1] * n_heads + [0] * n_tails
shuffle(tosses)

def plot_coins():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.bar(list(Counter(tosses).keys()), list(Counter(tosses).values()))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['tails', 'heads'])
    ax.set_ylim(0, 20)
    ax.set_yticks(np.arange(0, 21, 5))
    return fig

fig = plot_coins()
plt.show()
if __name__ == '__main__':
    with pm.Model() as coin_model:
        # Distributions are PyMC3 objects.
        # Specify prior using Uniform object.
        p_prior = pm.Uniform('p', 0, 1)

        # Specify likelihood using Bernoulli object.
        like = pm.Bernoulli('likelihood', p=p_prior, observed=tosses)     # "observed=data" is key
        # for likelihood.
        step = pm.Metropolis()

        # focus on this, the Inference Button:
        coin_trace = pm.sample(10000, step=step)
        pm.traceplot(coin_trace)
        plt.show()
        pm.plot_posterior(coin_trace[100:], color='#87ceeb',
                          rope=[0.48, 0.52], point_estimate='mean', ref_val=0.5)
        plt.show()

