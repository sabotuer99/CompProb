# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:23:52 2016

@author: TWhorten
"""
import numpy as np

observations = ["H", "H", "T", "T", "T"]
all_hstates = ["fair", "biased"]
all_obs = ["H", "T"]
prior = np.array([0.5, 0.5])
A = np.array([[0.75, 0.25],
              [0.25, 0.75]])
B = np.array([[0.5,0.5],
              [0.25, 0.75]])



log_prior = np.log2(prior)
log_a = np.log2(A)
log_b = np.log2(B)

print(log_a)
print(log_b)

m_12_fair = np.max(log_prior + log_a[0] + log_b[:,0])
m_12_bias = np.max(log_prior + log_a[1] + log_b[:,1])
m_12 = [m_12_fair, m_12_bias]
m_12_fair_argmax = np.argmax(log_prior + log_a[0] + log_b[:,0])
m_12_bias_argmax = np.argmax(log_prior + log_a[1] + log_b[:,1])
tail_12 = [m_12_fair_argmax, m_12_bias_argmax]
print(m_12)
print(tail_12)