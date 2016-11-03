# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:23:52 2016

@author: TWhorten
"""
import numpy as np

observations = ["H", "H", "T", "T", "T"]
all_hstates = ["fair", "bias"]
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

messages = np.array([[None] * len(all_hstates)] * (len(observations) + 1))
messages[0] = log_prior
back_pointers = np.array([[None] * len(all_hstates)] * (len(observations) + 1))

for i, obs in enumerate(observations):
    obs_i = all_obs.index(obs)    
    for j, state in enumerate(all_hstates):
       blarg = messages[i] + log_a[j] + log_b[:,obs_i]
       messages[i+1][j] = np.max(blarg) 
       back_pointers[i+1][j] = np.argmax(blarg)
"""        
    m_12_fair = np.max(log_prior + 
                       log_a[all_hstates.index("fair")] + 
                       log_b[:,obs_i])
    m_12_bias = np.max(log_prior + 
                       log_a[all_hstates.index("bias")] + 
                       log_b[:,obs_i])
    m_12 = [m_12_fair, m_12_bias]
    m_12_fair_argmax = np.argmax(log_prior + 
                                 log_a[all_hstates.index("fair")] + 
                                 log_b[:,all_obs.index(obs)])
    m_12_bias_argmax = np.argmax(log_prior + 
                                 log_a[all_hstates.index("bias")] + 
                                 log_b[:,all_obs.index(obs)])
    tail_12 = [m_12_fair_argmax, m_12_bias_argmax]
"""
print(messages)
print(back_pointers)

response = [None] * len(observations)
most_likely = np.argmax(messages[-1])
for i in range(len(back_pointers)-1, 0, -1):
    index = back_pointers[i][most_likely]
    most_likely = index
    response[i-1] = all_hstates[index]
    
print(response)
    


