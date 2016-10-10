# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
joint_prob_XY = np.array([[0.10, 0.09, 0.11], [0.08, 0.07, 0.07], [0.18, 0.13, 0.17]])

prob_X = joint_prob_XY.sum(axis=1)
prob_Y = joint_prob_XY.sum(axis=0)

joint_prob_XY_indep = np.outer(prob_X, prob_Y)

import scipy.stats as sp

print(sp)

D = sp.entropy(joint_prob_XY.flatten(), joint_prob_XY_indep.flatten(), 2)


print(D)