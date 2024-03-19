#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:15:17 2024

@author: solene
"""
###to be run after the joint diagonalization
###

from scipy.io import loadmat
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import os

condition_learn='E9WT'

current_path = os.getcwd()
path_input=current_path+'/input/'  
path_output=current_path+'/output/'

decomposition = loadmat(path_output+'V.mat')['V']
eigenvalues=loadmat(path_output+'all_diags.mat')['all_diags']

#####find the two highest components
nb_features=8
df_mean=pd.DataFrame(columns=['pc','eigen_mean'])
df_mean.pc=np.arange(nb_features)
df_mean.eigen_mean=np.mean(eigenvalues,axis=1).reshape(-1,1)
df_mean_sorted=df_mean.sort_values(by='eigen_mean', ascending=False).reset_index(drop=True)
pc1=int(df_mean_sorted.iloc[0].pc)
pc2=int(df_mean_sorted.iloc[1].pc)
#########################################"

#############make pc coordinates for all embryos######

def make_coord(df_embryo,e):
    df_f=df_embryo.drop(['id_cell','condition','embryo'],axis=1)
    X=np.array(df_f)
    PC_coord=X@(decomposition)
    myPC_coord1=PC_coord[:,pc1]
    myPC_coord2=PC_coord[:,pc2]
    vectors1=myPC_coord1.reshape(-1,1)
    vectors2=myPC_coord2.reshape(-1,1)
    df_vectors=pd.DataFrame(columns=['id_cell','pc_coord1','pc_coord2'])
    df_vectors['id_cell']=df_embryo.id_cell
    df_vectors['pc_coord1']=vectors1
    df_vectors['pc_coord2']=vectors2
    return df_vectors


df_vectors_all=pd.DataFrame()
df_features=pd.read_pickle(path_input+'df_features_scaled.pkl')
e_ids=list(np.unique(df_features.embryo))

for e in e_ids:
    df_embryo=df_features.loc[df_features['embryo']==e].reset_index(drop=True)
    df_vectors=make_coord(df_embryo,e)
    df_vectors['condition']=df_embryo.condition[0]
    df_vectors['embryo']=e
    df_vectors_all=pd.concat([df_vectors_all, df_vectors])
    
df_vectors_all.to_pickle(path_output+'df_vectors_all.pkl')

###best number of clusters according to aic and bic score
df_learn=df_vectors_all.loc[df_vectors_all['condition']==condition_learn]
tmp_list=[]
for nc in range(1,10):
    n=len(df_learn)
    vectors_learn=np.array((df_learn['pc_coord1']))
    gm = GaussianMixture(n_components=nc).fit(vectors_learn.reshape(-1,1))
    my_labels=gm.predict(vectors_learn.reshape(-1,1))
    gm_aic=gm.aic(vectors_learn.reshape(-1,1))
    gm_bic=gm.bic(vectors_learn.reshape(-1,1))
    print(nc)
    print('aic '+str(gm_aic))
    print('bic '+str(gm_bic))
    row={'n':nc, 'aic':gm_aic,'bic':gm_bic}
    tmp_list.append(row)
df_gm_scores=pd.DataFrame(tmp_list)
df_gm_scores.to_pickle(path_output+'df_gm_scores.pkl')



####chose nc manually there according to aic and bic criteria
nc=3

####robustness of gaussian mixture
nb_iter=100
labels_iter=np.array([[0]*nb_iter]*n)

for i in range(nb_iter):
    print('clustering '+str(i))
    vectors_all=np.array((df_learn['pc_coord1']))
    gm = GaussianMixture(n_components=nc).fit(vectors_all.reshape(-1,1))
    my_labels=gm.predict(vectors_all.reshape(-1,1))
    labels_iter[:,i]=my_labels

M = np.array([[0]*n]*n)
for i in range(nb_iter):
    print('check '+str(i))
    df_labels = pd.DataFrame(columns=['label'])
    df_labels.label = labels_iter[:, i]
    for c in range(nc):
        this_cluster = df_labels.loc[df_labels['label'] == c]
        this_cluster_cells = list(this_cluster.index)
        for i1 in range(len(this_cluster_cells)):
            c1 = this_cluster_cells[i1]
            for i2 in range(i1, len(this_cluster_cells)):
                c2 = this_cluster_cells[i2]
                M[c1, c2] = M[c1, c2]+1

values = M[np.triu_indices(n)]
values = np.divide(values, nb_iter)
uncertain = [i for i in values if i > 0 and i < 1]
uncertain_pct = (len(uncertain)/len(values))*100
print(str(uncertain_pct)+' percent of the pairs of nodes are not either always or never clustered together')







