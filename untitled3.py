# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 09:40:10 2016

@author: TWhorten
"""
from simpsons_paradox_data import *

for x in ['A', 'B', 'C', 'D', 'E', 'F']:
    female_and_X_only = joint_prob_table[gender_mapping['female'], 
                                         department_mapping[x]]
    prob_admission_given_female_and_X = female_and_X_only / np.sum(female_and_X_only)
    prob_admission_given_female_and_X_dict = dict(zip(admission_labels, prob_admission_given_female_and_X))
    print('Female admission rate ({0}): {1:.5}'.format(x, prob_admission_given_female_and_X_dict['admitted']))
    
    male_and_X_only = joint_prob_table[gender_mapping['male'], 
                                         department_mapping[x]]
    prob_admission_given_male_and_X = male_and_X_only / np.sum(male_and_X_only)
    prob_admission_given_male_and_X_dict = dict(zip(admission_labels, prob_admission_given_male_and_X))
    print('Male admission rate ({0}): {1:.5}'.format(x, prob_admission_given_male_and_X_dict['admitted']))
    