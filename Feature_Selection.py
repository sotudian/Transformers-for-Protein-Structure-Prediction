#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: shahab Sotudian
"""


import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")
import scipy.io
import random
random.seed(2)
from sklearn.metrics import confusion_matrix
import csv
import pickle
import numpy as np
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None
from numpy import save
from numpy import load

# Functions  ###################=================----------------------

map_folder='./Project/features/att_map_esm/'
label_folder='./Project/lables/dist_map/'

def Att_map_Data_Generation(att_map,label_map):
    att_map=att_map.transpose(0,2,3,1)[0,:,:,:]
    att_map=att_map[1:-1,1:-1,:]
    Data_Att_map_X = []
    Data_Att_map_Y = []
    for i in range(att_map.shape[0]):
        for j in range(i+1,att_map.shape[1]):
            P1 = list(att_map[i,j,:])
            Data_Att_map_X.append(P1)
            Data_Att_map_Y.append(label_map[i,j])
    return np.array(Data_Att_map_X),np.array(Data_Att_map_Y)



def Prediction_Func(Test_X,reg):
    att_map=Test_X.transpose(0,2,3,1)[0,:,:,:]
    att_map=att_map[1:-1,1:-1,:]
    Predicted_Att_map = np.ones((att_map.shape[0],att_map.shape[1]))*100
    for i in range(att_map.shape[0]):
        for j in range(i+1,att_map.shape[1]):
            P1= reg.predict(np.array(att_map[i,j,:]).reshape(1, -1))
            Predicted_Att_map[i,j] = Predicted_Att_map[j,i] = P1
    return Predicted_Att_map         

def score(distance_mat, predicted_mat):
    L = len(predicted_mat)
    results = []
    for seq_sep in [5, 24]:
        for num_top in [5, 10, int(L/5), int(L/2), int(L)]: 
            assert len(distance_mat) == len(predicted_mat)
            #L = len(predicted_mat)
            
            indices_upper_tri = np.triu_indices(L, seq_sep)
                    
            df_data = pd.DataFrame()
            df_data['residue_i'] = indices_upper_tri[0] + 1
            df_data['residue_j'] = indices_upper_tri[1] + 1
            df_data['confidence'] = predicted_mat[indices_upper_tri]
            
            df_data['distance'] = distance_mat[indices_upper_tri]
            df_data['contact'] = ((df_data.distance < 8) * 1).tolist()
               
            df_data.sort_values(by='confidence', ascending=False, inplace=True)
                    
            sub_true = (df_data.query('distance > 0').head(num_top).query('contact > 0')).shape[0]
            sub_false = (df_data.query('distance > 0').head(num_top).query('contact < 1')).shape[0]
            
            precision = 100 * sub_true / (sub_true + sub_false)
            results.append([seq_sep, num_top, precision])
    df_results = pd.DataFrame(data = results, columns = ['seq_sep', 'num_top', 'precision'])
               
    return df_results  



# Model Testing  ###################=================----------------------

Flag = 0

if Flag:
    Testing_List = list(os.listdir('./Project/Featureselection_Proteins'))
    Layer_SCORES = np.zeros(660)
    for t,i in enumerate(Testing_List):
        print(t+1,') Sequences:',i,':')     
        Test_X = np.load('./Project/TESTING/TEST_Att/'+Testing_List[t])
        Test_Y_Truth = np.load('./Project/TESTING/TEST_dist_map/TEST_dist_map/'+Testing_List[t])
        Performance_FB = pd.read_csv('./Project/TESTING/score_esm/score_esm/'+str(Testing_List[t][:-3])+'csv')   
        att_map=Test_X.transpose(0,2,3,1)[0,:,:,:]
        att_map=att_map[1:-1,1:-1,:]
        for L in range(660):
            One_att_map = att_map[:,:,L] # just 1
            Layer_SCORES[L] += score(Test_Y_Truth, One_att_map)['precision'][2] # L/5
    # save to npy file
    Layer_SCORES = Layer_SCORES/len(Testing_List)
    save('Heads_Scores.npy', Layer_SCORES)
else:
     # load array
    Layer_SCORES = load('Heads_Scores.npy')


# Plotting   ###################=================----------------------

import matplotlib.pyplot as plt
plt.style.use('ggplot')

Testing_List = [i for i in range(32)]

x = range(len(Layer_SCORES))
energy = Layer_SCORES/len(Testing_List)

x_pos = [i for i, _ in enumerate(x)]
plt.figure(dpi=300)
plt.bar(x_pos, energy, color='blue')

plt.xlabel("660 Layers")
plt.ylabel("Performance - L/5")
plt.xaxis.set_visible(False)

plt.xticks(x_pos, x)

plt.show()
plt.savefig('Perf_Heads.png')




