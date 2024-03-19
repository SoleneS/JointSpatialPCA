#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:08:56 2024

@author: solene
"""
###create folders input and output in the same folder as this script
###in the input folder put 
###1)all the adjacency matrices as dataframes using the cells names cell_id as indices and columns names
#### the adjacency matrices should be named 'Ae'+str(e_id) for each e_id
###2)the dataframe of all centered and scaled features, called 'df_features_scaled.pkl'
### with a column 'embryo' for e_id of the embryos 
###and a column 'id_cell' that correspond the ids in adjacency dataframes
###and a column 'condition"
###all the other columns should be the features values

####OUTPUT: a concatenation of the covariance matrices





import pandas as pd
import numpy as np
import copy
import os
from scipy.io import savemat

def make_covariance(e):
    ###features
    df_features_embryo=df_features_scaled.loc[df_features_scaled['embryo']==e].reset_index(drop=True)
    retained_cells=list(df_features_embryo.id_cell)
    ####connect
    df_A=pd.read_pickle(path_input+'Ae'+str(e)+'.pkl')
    all_cells=list(df_A.index)
    df_old_A=copy.deepcopy(df_A)
    ####remove the cells with nan values
    cells_not_retained=list(set(all_cells)-set(retained_cells))
    df_A=df_A.drop(columns=cells_not_retained)
    df_A=df_A.drop(cells_not_retained)
    A=np.array(df_A)
    sumA=np.sum(A,axis=1)
    i_0=np.where(sumA==0)[0]
    ind_0=list(df_A.index[i_0])
    while len(ind_0)>0:
        print(len(ind_0))
        ####remove the newly disconnected nodes
        df_A=df_A.drop(columns=ind_0)
        df_A=df_A.drop(ind_0)
        ####remove newly disconnected from df_features
        df_features_embryo=df_features_embryo.drop(df_features_embryo[df_features_embryo['id_cell'].isin(ind_0)].index)
        A=np.array(df_A)
        sumA=np.sum(A,axis=1)
        i_0=np.where(sumA==0)[0]
        ind_0=list(df_A.index[i_0])
    ####make M
    L=A/sumA[:,None]
    ####make X
    df_f=df_features_embryo.drop(['id_cell','condition','embryo'],axis=1)
    X=np.array(df_f)
    ####make matrix to diagonalise
    n=len(df_f)
    M=np.transpose(X)@(L+np.transpose(L))@X
    M=M/(2*n)
    return M




# Construct a path relative to the script's location
current_path = os.getcwd()
path_input=current_path+'/input/'  
path_output=current_path+'/output/'


df_features_scaled=pd.read_pickle(path_input+'df_features_scaled.pkl')
e_ids=[1,2,3,4,5,6]
n_features=8

M_cat=np.empty((n_features,n_features)) ###random init
df_features_all=pd.DataFrame()
i=0

for e in e_ids:
    M=make_covariance(e)
    M_cat=np.hstack((M_cat,M))
    i=i+1
    
M_cat=M_cat[:,n_features:] ###remove the random one that was used to initialize


savemat(path_output+'M_cat',{'M_cat':M_cat})

