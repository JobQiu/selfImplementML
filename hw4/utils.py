import scipy.io
import numpy as np
import pandas as pd
from sklearn import preprocessing

#### Implement the Gaussian kernel here ####

def gaussian_kernel(x1,x2,sigma):
    return np.exp(-np.sum((x1-x2)**2) / (2. * sigma**2))

def distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

#### End of your code ####

# load a mat file


def load_mat(fname):
    data = scipy.io.loadmat(fname)
    X = data['X']
    y = data['y'].flatten()
    return X,y

def loadval_mat(fname):
    data = scipy.io.loadmat(fname)
    X = data['X']
    y = data['y'].flatten()
    X_val = data['Xval']
    y_val = data['yval'].flatten()
    return X,y, X_val, y_val

# plot training data

def get_vocab_dict():
    words = {}
    inv_words = {}
    f = open('data/vocab.txt','r')
    for line in f:
        if line != '':
            (ind,word) = line.split('\t')
            words[int(ind)] = word.rstrip('\n')
            inv_words[word.rstrip('\n')] = int(ind)
    return words, inv_words
