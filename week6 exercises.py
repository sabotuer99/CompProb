# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 08:21:49 2016

@author: TWhorten
"""
import numpy as np
from sklearn.preprocessing import normalize

psi_12 = np.array([
        [5, 1],
        [1, 5],
    ])
    
psi_23 = np.array([
        [0, 1],
        [1, 0],
    ])

n1 = psi_12 / psi_12.sum() #normalizes the whole array
n2 = psi_12 / psi_12.sum(axis=0) #normalizes the columns
n3 = psi_12 / psi_12.sum(axis=1) #normalizes the rows?
n4 = normalize(psi_12.astype(np.float64), axis=1, norm="l1")


#print(psi_12)
#print(n1)
#print(n2)
#print(n3)
print(n4[0].reshape(2,1)) #this is the one I need

print(n4[0].reshape(2,1) * psi_23)

print(np.matmul(psi_12, psi_23))

