#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 13:35:15 2018

@author: xavier.qiu
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from nn import *

np.random.seed(1) # set a seed so that the results are consistent

#%%

X = np.array([[1,2,3,4],[5,6,7,8]])
y = np.array([1,0])
sizes = layer_sizes(X,y,[2,3])
