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
from numpy import load
import math
import numpy as np
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_rows = None
import os
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb   
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
# import pylab as pl
import json
import copy

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

def my_train(X_train, y_train, model='LR', penalty='l1', cv=5, scoring='f1', class_weight= 'balanced',seed=2020):    
    if model=='SVM':
        svc=LinearSVC(penalty=penalty, class_weight= class_weight, dual=False, max_iter=10000)#, tol=0.0001
        parameters = {'C':[0.001,0.01,0.1,1,10,100,1000]} #'kernel':('linear', 'rbf'), 
        gsearch = GridSearchCV(svc, parameters, cv=cv, scoring=scoring) 
        
    elif model=='LGB':        
        param_grid = {
            'num_leaves': range(2,15,4),
            'n_estimators': [50,100,500,1000],
            'colsample_bytree': [ 0.1,0.5, 0.9]#[0.6, 0.75, 0.9]

            }
        lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt',  objective='binary', learning_rate=0.1, random_state=seed)# eval_metric='auc' num_boost_round=2000,
        gsearch = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid, cv=cv,n_jobs=-1, scoring=scoring)
    elif model=='RF': 
        rfc=RandomForestClassifier(random_state=seed, class_weight= class_weight, n_jobs=-1)
        param_grid = { 
            'max_features':[0.05,0.3, 0.5, 0.7, 1],#, 0.4, 0.5, 0.6, 0.7, 0.8 [ 'sqrt', 'log2',15],#'auto'  1.0/3,
            'n_estimators': [50, 100,500,1000],
            'max_depth' : range(2,10,1)#[2, 10]
                        
        }
        gsearch = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=cv, scoring=scoring)    
     
    else:
        LR = LogisticRegression(penalty=penalty, class_weight= class_weight,solver='liblinear', random_state=seed)
        parameters = {'C':[0.01,0.1,1,10,100,1000] } 
        gsearch = GridSearchCV(LR, parameters, cv=cv, scoring=scoring) 
        # clf = LogisticRegressionCV(Cs=[10**-1,10**0, 10], penalty=penalty, class_weight= class_weight,solver='liblinear', cv=cv, scoring=scoring, random_state=seed)#, tol=0.01
    gsearch.fit(X_train, y_train)
    clf=gsearch.best_estimator_
    if model=='LGB' or model=='RF': 
        print('Best parameters found by grid search are:', gsearch.best_params_)
    print('train set accuracy:', gsearch.best_score_)
    return clf

def Prediction_Func(Test_X,reg,TOP_Num):
    att_map=Test_X.transpose(0,2,3,1)[0,:,:,:]
    att_map=att_map[1:-1,1:-1,:]
    att_map = att_map[:,:,(660-TOP_Num):]
    
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




# Model Training  ###################=================----------------------

Training_Flag = 1
TOP_Num = 20  # Top 20

# load array
Layer_SCORES = load('Heads_Scores.npy')

SCORES_Layers = np.vstack([np.array(list(range(len(Layer_SCORES)))),Layer_SCORES]).T
SCORES_Layers = pd.DataFrame(SCORES_Layers, columns = ['Num','Score'])
SCORES_Layers = SCORES_Layers.sort_values(by=['Score'], ascending=False)
SCORES_Layers.reset_index(inplace = True, drop=True)

# Best Layers

Top_Layers = list(SCORES_Layers['Num'][:TOP_Num]) 
Top_Layers = [int(i) for i in Top_Layers]


# Data Generation ###################=================----------------------
if Training_Flag:
    X = np.empty((0,len(Top_Layers)), int)
    Y = np.empty((0,), int)
    Training_List = list(os.listdir(map_folder))
    
    for t,i in enumerate(Training_List):
        print(t+1,') Sequences:',i)
        label_map=np.load(label_folder+Training_List[t])
        att_map=np.load(map_folder+Training_List[t])
        Data_Att_map_X,Data_Att_map_Y = Att_map_Data_Generation(att_map,label_map)
        X = np.append(X,Data_Att_map_X[:,Top_Layers], axis=0)   # just Top_Layers     
        Y = np.append(Y,Data_Att_map_Y, axis=0)  
    del Data_Att_map_X,Data_Att_map_Y,att_map,label_map,t
    Y_Classification = np.where(Y < 8, 1, 0)




if Training_Flag:
    MODEL = LinearRegression().fit(X, Y)
    pickle.dump(MODEL, open('Reg_OLS_Top20.sav', 'wb'))            
else:
    MODEL = pickle.load(open('Reg_OLS_Top20.sav', 'rb'))



# Model Testing  ###################=================----------------------



Testing_List_Att = list(os.listdir('./Project/TESTING/TEST_Att'))
Testing_List_Dist = list(os.listdir('./Project/TESTING/TEST_dist_map/TEST_dist_map'))


Testing_List = ['5ZX9_A.npy','6IEH_A.npy','6DRF_A.npy','5OQK_A.npy','6D9F_B.npy','6HC2_X.npy',
                '6EAZ_B.npy','5Z6D_B.npy','6D0I_C.npy','5YVQ_B.npy','6FXD_B.npy','6E9O_A.npy',
                '6D7Y_B.npy','5YVQ_A.npy','6D0I_D.npy','6DGN_B.npy','5ZB2_A.npy','6IAI_D.npy',
                '6CP9_G.npy','5OD1_A.npy','6CZ6_D.npy','5Z9T_B.npy','5OVM_A.npy','6G3B_B.npy',
                '6E0K_A.npy','5ZKH_B.npy','6CMK_A.npy','5YA6_B.npy','6GHO_B.npy','5W5P_A.npy',
                '6H2X_A.npy','6CP8_D.npy','5ZT7_B.npy','5ZER_B.npy','5ZKE_A.npy','6FCG_F.npy',
                '6FTO_C.npy','6A5F_B.npy','6I9H_A.npy','5ZYO_D.npy','6CZT_A.npy','6E3C_C.npy',
                '6A2W_A.npy','5ZKT_B.npy','6DFL_A.npy','6A9J_B.npy','6CGO_B.npy','6D2S_A.npy',
                '6BZK_A.npy','6NU4_A.npy']


for t,i in enumerate(Testing_List):
    print(t+1,') Sequences:',i,':')     
    Test_X = np.load('./Project/TESTING/TEST_Att/'+Testing_List[t])
    Test_Y_Truth = np.load('./Project/TESTING/TEST_dist_map/TEST_dist_map/'+Testing_List[t])
    Performance_FB = pd.read_csv('./Project/TESTING/score_esm/score_esm/'+str(Testing_List[t][:-3])+'csv')   
    Predicted_Att_Map = Prediction_Func(Test_X,MODEL,TOP_Num)  
    Score_Predict = score(Test_Y_Truth, Predicted_Att_Map)
    Score_Predict['precision_FB'] = list(Performance_FB['precision'])
    print(Score_Predict)
    if t == 0:
        All_Score_Predict = Score_Predict
    else:
        All_Score_Predict += Score_Predict
    


All_Score_Predict = All_Score_Predict/len(Testing_List)
