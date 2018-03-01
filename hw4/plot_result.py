#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 13:09:46 2018

@author: xavier.qiu
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
sns.set()


plt.figure(figsize=(18,12))
num_iters = 2000
#loss_ = np.load("simga_210.0C_0.3learn5000000.npy")
#plt.plot(loss_,alpha=0.3)
#loss_ = np.load("simga_210.0C_0.3learn1000000.npy")
#plt.plot(loss_,alpha=0.5)
loss_ = np.load("simga_210.0C_0.3learn100000.npy")
temp_LR = linear_model.LinearRegression()
XXX = np.array(range(num_iters-100)).reshape(num_iters-100,1)
yyy = np.array(loss_)[100:]
plt.plot(loss_)
loss_ = np.load("simga_210.0C_0.3learn10000.npy")
plt.plot(loss_)
loss_ = np.load("simga_210.0C_0.3learn1000.npy")
plt.plot(loss_)
loss_ = np.load("simga_210.0C_0.3learn100.npy")
plt.plot(loss_)
loss_ = np.load("simga_210.0C_0.3learn10.npy")
plt.plot(loss_)
loss_ = np.load("simga_210.0C_0.3learn1.npy")
plt.plot(loss_)
#plt.ylim(0.15,0.35)
#%%

