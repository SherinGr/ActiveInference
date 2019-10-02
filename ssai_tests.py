# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:59:09 2019

@author: Sherin
"""
import numpy as np
import sys
sys.path.insert(0, r'C:\Users\sheri\Desktop\Sherins Map All Things\Research\Active Inference\SSAI\active_inference')
import ssai
from scipy.linalg import toeplitz
#import active_inference.ssai as ss

# File to test stuff:

mu = np.matrix([[1,2,3],[1,3,4]])

check3 = (mu-2*mu).dot(mu.T)

mu2= np.matrix(mu)

for x in mu2.shape:
    if x != 4:
        N = x

#muvec = np.reshape(mu,[1,np.size(mu)])
muvec = np.matrix.flatten(mu)
check = muvec.T

check2 = np.dot(muvec,muvec.T)

k = 4
c = np.zeros([1,k])
r = np.array([0,1])
r2 = np.append(r,np.zeros([1,k-2]))

T = toeplitz(np.zeros([1,k]),np.append(np.array([0,1]),np.zeros([1,k-2])))

dims = np.shape(mu)

mins = min(dims)

if mins == 1:
    raise ValueError('OOPS')
    
class test():
    def this(self,y):
        return y+2
    
temp = test()
z = temp.this(3)

ssai.makeNoise()

