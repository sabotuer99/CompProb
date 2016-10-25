# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:19:30 2016

@author: TWhorten
"""
import numpy as np

phi_1 = np.array([0.5, 0.5]).reshape(2,1)
phi_2 = np.array([0.5, 0.5]).reshape(2,1)
phi_3 = np.array([0.6, 0.4]).reshape(2,1)
phi_4 = np.array([0.8, 0.2]).reshape(2,1)
phi_5 = np.array([0.8, 0.2]).reshape(2,1)

psi = np.array([[0.0, 1.0],
                [1.0, 0.0]])
                
m_52 = (phi_5 * psi).sum(axis=0)
m_42 = (phi_4 * psi).sum(axis=0)
m_31 = (phi_3 * psi).sum(axis=0)
m_21 = (phi_2 * psi * m_52 * m_42).sum(axis=1)
m_12 = (phi_1 * psi * m_31).sum(axis=1)   
px2 = (phi_2 * m_12).sum(axis=0)

print(px2 / np.sum(px2))
print(m_12 / np.sum(m_12))              

#part (d)
phi_3 = np.array([0.0, 1.0]).reshape(2,1)
phi_4 = np.array([1.0, 0.0]).reshape(2,1)
phi_5 = np.array([0.0, 1.0]).reshape(2,1)
psi = np.array([[0.2, 0.8],
                [0.8, 0.2]])

m_52 = (phi_5 * psi).sum(axis=0)
m_42 = (phi_4 * psi).sum(axis=0)
m_31 = (phi_3 * psi).sum(axis=0)
m_21 = (phi_2 * psi * m_52 * m_42).sum(axis=1)
m_12 = (phi_1 * psi * m_31).sum(axis=1)   
px2 = (phi_2 * m_12).sum(axis=0)

print("part (d)")
print(m_42 / np.sum(m_42))
print(m_52 / np.sum(m_52))
print(m_31 / np.sum(m_31))
print(m_12 / np.sum(m_12))
print(px2 / np.sum(px2))





