#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:09:10 2024

@author: solene
"""


from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import os

current_path = os.getcwd()
path_input=current_path+'/input/'  
path_output=current_path+'/output/'

decomposition = loadmat(path_output+'V.mat')['V']
eigenvalues=loadmat(path_output+'all_diags.mat')['all_diags']



df_vectors_all=pd.read_pickle(path_output+'df_vectors_all.pkl')


nc=3
condition_learn='E9WT'
df_all_labels=pd.DataFrame()


####train GM on E9WT data
df_learn=df_vectors_all.loc[df_vectors_all['condition']==condition_learn].reset_index(drop=True)
vectors_learn=np.array((df_learn.pc_coord1))
X=vectors_learn.reshape(-1,1)
gmm=GaussianMixture(n_components=3,random_state=0)
gmm.fit(X)
probabilities=gmm.predict_proba(X)
labels=gmm.predict(X)
df_learn['label']=labels

df_all_labels=pd.concat([df_all_labels, df_learn])

gm_aic=gmm.aic(X)
gm_bic=gmm.bic(X)
print('aic '+str(gm_aic))
print('bic '+str(gm_bic))

gm_means=gmm.means_
gm_stds=gmm.covariances_
gm_weights = gmm.weights_


plt.hist(X, density=True, alpha=0.7, color='blue', label='PC1')
x_range = np.linspace(X.min(), X.max(), 1000)
from scipy.stats import norm
for i in range(len(gm_means)):
    mean, cov, weight = gm_means[i, 0], gm_stds[i, 0, 0], gm_weights[i]
    curve = norm.pdf(x_range, mean, np.sqrt(cov)) * weight
    plt.plot(x_range, curve, label=f'Gaussian {i + 1}')

plt.title('Histogram with Gaussian Curves '+ condition_learn)
plt.xlabel('PC1')
plt.ylabel('Density')
plt.legend()
#plt.show()
plt.savefig(path_output+'histogram_pc1_gm_'+condition_learn+'.png')
plt.close()

all_conditions=list(np.unique(df_vectors_all.condition))
all_conditions.remove(condition_learn)

for condition in all_conditions:
    df_condition=df_vectors_all.loc[df_vectors_all['condition']==condition].reset_index(drop=True)
    vectors_condition=np.array(df_condition.pc_coord1)
    X_new=vectors_condition.reshape(-1,1)
    new_probabilities=gmm.predict_proba(X_new)
    new_labels=gmm.predict(X_new)
    df_condition['label']=new_labels
    df_all_labels=pd.concat([df_all_labels,df_condition])
    plt.hist(X_new, density=True, alpha=0.7, color='blue', label='PC1')
    x_range = np.linspace(X_new.min(), X_new.max(), 1000)
    from scipy.stats import norm
    for i in range(len(gm_means)):
        mean, cov, weight = gm_means[i, 0], gm_stds[i, 0, 0], gm_weights[i]
        curve = norm.pdf(x_range, mean, np.sqrt(cov)) * weight
        plt.plot(x_range, curve, label=f'Gaussian {i + 1}')
    plt.title('Histogram '+condition+' with Gaussian Curves learned on '+condition_learn)
    plt.xlabel('PC1')
    plt.ylabel('Density')
    plt.legend()
    #plt.show()
    plt.savefig(path_output+'histogram_pc1_gm_'+condition+'.png')
    plt.close()
   

df_all_labels.to_pickle(path_output+'df_labels.pkl')

