'''
Created on Feb 15, 2017

@author: root
'''

import numpy as np
import math

def sigmoid(z):
    try:
        return 1 / (1 + math.exp(-z))    
    except:
        if z > 6:
            return 0.99
        return 0.001
        
  
  
def rectifiedLinear(z):
    if z <= 0:
        return 0.0
    elif z < 1 and z>0:
        return z
    else:
        return 1     


def arcTan(z):
    return np.arctan(z)


def tanH(z):
    return np.tanh(z)

def ttt(z): 
    try:
        return 1 / (1 + math.exp(-z))    
    except:
        if z > 6:
            return 1.0
        return 0.0 
