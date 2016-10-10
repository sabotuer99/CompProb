# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 13:33:11 2016

@author: TWhorten
"""
import numpy as np

prob_W_I = np.array([[1/2, 0], [0, 1/6], [0, 1/3]])

prob_W = prob_W_I.sum(axis=1)
prob_I = prob_W_I.sum(axis=0)

print(np.outer(prob_W, prob_I))