#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 13:32:19 2018

@author: xavier.qiu
"""

def layer_sizes(X, Y, hidden_layer_sizes):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_y -- the size of the output layer
    """
    ### START CODE HERE ### (≈ 3 lines of code)
    n_x = X.shape[0] # size of input layer 
    n_y = Y.shape[0]# size of output layer
    ### END CODE HERE ###
    
    return (n_x,hidden_layer_sizes, n_y)