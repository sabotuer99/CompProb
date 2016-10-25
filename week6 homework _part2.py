# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:27:22 2016

@author: TWhorten
"""
import numpy as np

#Harry Potter part
prior = np.array([0.5, 0.5]).reshape(2,1)

phi_1 = prior
phi_2 = prior
phi_3 = prior
phi_6 = prior
phi_7 = prior
phi_5 = prior
phi_4 = np.array([1, 0]).reshape(2,1)

psi = np.array([[0.5, 0.5],
                [0.25, 0.75]])
                
m_52 = (phi_5 * psi).sum(axis=0).reshape(2,1)
m_42 = (phi_4 * psi).sum(axis=0).reshape(2,1)
m_21 = (phi_2 * psi * m_52 * m_42).sum(axis=0)

m_63 = (phi_6 * psi).sum(axis=0).reshape(2,1)
m_73 = (phi_7 * psi).sum(axis=0).reshape(2,1)
m_31 = (phi_3 * psi * m_63 * m_73).sum(axis=1)

print(m_21 / np.sum(m_21))
print(m_31 / np.sum(m_31))

px1 = (phi_1 * m_21 * m_31).sum(axis=0)
print(px1 / np.sum(px1))