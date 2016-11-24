# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 08:13:19 2016

@author: twhorten
"""
import numpy as np
from math import log

obs = np.array([
[1,1,1,0,0],
[1,1,0,1,0],
[1,1,1,1,1],
[1,0,1,0,0],
[1,1,1,0,0],
[1,0,1,0,0],
[0,0,0,0,0]])

tots = obs.sum(axis=0)

print(tots)

lus = [0] * 4
print(lus)

for n in range(4):
    lu = 0
    for i in range(7):
        if obs[i][n] == 1:
            if obs[i][4] == 1:
                lu += log(0.25)
            else:
                lu += log(0.75)
    lus[n] = lu

print(lus)