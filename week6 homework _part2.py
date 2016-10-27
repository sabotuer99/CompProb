# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:27:22 2016

@author: TWhorten
"""
import numpy as np

#Harry Potter part
prior = np.array([1,1]).reshape(2,1)

phi_1 = prior
phi_2 = prior
phi_3 = prior
phi_6 = prior
phi_7 = prior
phi_5 = prior
phi_4 = np.array([1, 0])

psi = np.array([[0.5, 0.5],
                [0.25, 0.75]])
                
m_52 = (phi_5 * psi).sum(axis=1).reshape(2,1)
m_42 = (phi_4 * psi).sum(axis=1).reshape(2,1)

factor1 = (phi_2 * m_52 * m_42).reshape(1,2)
#print(factor1)

m_21 = (psi * factor1).sum(axis=1).reshape(2,1)
print(m_21)

#print(phi_2)
#print(psi)
#print(m_52)
#print(m_42)

m_63 = (phi_6 * psi).sum(axis=1).reshape(2,1)
m_73 = (phi_7 * psi).sum(axis=1).reshape(2,1)
#print(m_63)
#print(m_73)
factor2 = (phi_3 * m_63 * m_73).reshape(1,2)
#print(factor2)
#print(psi)
m_31 = (psi * factor2).sum(axis=1).reshape(2,1)

#print("######")

print(m_31)

px1 = (phi_1 * m_21 * m_31)
print(px1 / np.sum(px1))