#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 12:39:01 2023

@author: jonas
"""

import h5py
import numpy as N
import os
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import confusion_matrix, mean_absolute_error

target_dir="/Users/jonas/Documents/projects/rest_move_sequences/results_files" # Directory in which the results file is saved.
data_dir="/Users/jonas/Documents/projects/rest_move_sequences/data/hdf5" # Adjust to data location

# Params
minima_search_range=10 # Max frames to detect a local minimum following sampling zone entry = true sampling start.
n_frames=20
n_neurons=100

start_point='samp' # Which variable to use. Options are 'samp', 'oc', 'os', 'rew'

n_splits=5
C=3
max_iter=1000000000
tol=0.000000000001
iterations=100
roll_range=range(40,400,1) # Range of time-shift for the signal in X_shuff (in dp). The time shift is performed once on the raw data.


# Model.
clf=SVC(C=C,max_iter=max_iter,tol=tol,kernel='linear')
kf=KFold(n_splits=n_splits,shuffle=True)

# Seeding.
N.random.seed(42)

# Helper functions.
def separate_in_2d_array(data,manual_in):

    start=manual_in[0::2]
    end=manual_in[1::2]
    
    extracted=[]
    for i in range(len(data)):
        for n in range(len(start)):
            extracted.extend(data[i][int(start[n]):int(end[n])])
    extracted=N.reshape(extracted,[len(data),-1])
    return extracted


def separate_in(indices,not_in):
    ''' Separate 2 periods of spike-like (point) data.
    '''    
   
    not_in_range=[]
    for n in range(len(not_in[0::2])):
        not_in_range.extend(range(not_in[0::2][n],not_in[1::2][n],1))
    not_in_range=set(not_in_range)
  
    ind=[]
    for n in range(len(indices)):
        if indices[n] in not_in_range:
            ind.append(indices[n])
   
    return ind

def get_X_y(filename,session):
    f = h5py.File(filename, 'r')
    
    spks=N.asarray(f['%s/analysis/trace/Fn' %session])
    track=N.asarray(f['%s/behavior/track'%session])

        
    sampl=N.asarray(f['%s/behavior/%sl' %(session,start_point)])
    sampr=N.asarray(f['%s/behavior/%sr' %(session,start_point)])
    corr=N.asarray(f['%s/behavior/corr'%session])
    
    sampl_raw=separate_in(sampl,corr)
    sampr_raw=separate_in(sampr,corr)
    
    # Correct for odour sampling onset.
    
    sampl,sampr=[],[]
    y_raw=track[1]
    for n in range(len(sampl_raw[0::2])):
        local=sampl_raw[0::2][n]
        offset=N.argmin(y_raw[local:local+minima_search_range])
        sampl.append(local+offset)
    for n in range(len(sampr_raw[0::2])):
        local=sampr_raw[0::2][n]
        offset=N.argmin(y_raw[local:local+minima_search_range])
        sampr.append(local+offset)
    sampl=N.asarray(sampl)
    sampr=N.asarray(sampr)
    
    
    
    def get_start_end(samp):
        points,y=[],[]
        for n in range(len(samp[0::2])):
            start=samp[0::2][n]
            end=samp[0::2][n]+n_frames
            
            y.extend(range(0,end-start,1))
    
            points.append(start)
            points.append(end)
        return y,points
    
    y_left,points_left=get_start_end(sampl)
    y_right,points_right=get_start_end(sampr)
    
    X_left=separate_in_2d_array(spks,points_left)
    X_right=separate_in_2d_array(spks,points_right)
    
    # Compensate for unequal left-right trial counts.
    
    if len(y_left)>len(y_right):
        y_left=y_left[:len(y_right)]
        X_left=X_left[:,:len(y_right)]
    elif len(y_left)<len(y_right):
        y_right=y_right[:len(y_left)]
        X_right=X_right[:,:len(y_left)]
    
    
    
    X=N.append(X_left,X_right,axis=1)
    y=N.append(y_left,y_right)

        
    return X,y,X_left,y_left,X_right,y_right

# Executed part.
os.chdir(data_dir)
filename_list=['gcamp6f044_learned.hdf5',
               'gcamp6f094_learned.hdf5',
               'gcamp6f216_learned.hdf5',
               'gcamp6f218_learned.hdf5',
               'gcamp6f219_learned.hdf5',
               'gcamp6f478_learning.hdf5',
               'gcamp6f481_learning.hdf5',
               'gcamp6f483_learning.hdf5',
               'gcamp6f485_learning.hdf5']
              
session_list=[0,0,0,0,0,2,2,2,2]    
animal_list=["44","94","216","218","219","478","481","483","485"]
                      
n_reg=1
    
score_total=N.empty((len(animal_list)))
score_shuff_total=N.empty_like(score_total)
cm_total,cm_shuff_total=[],[]

for animal in range(len(animal_list)):
    filename=filename_list[animal]
    name=animal_list[animal]
    session=session_list[animal]
    print(name)
        
    X,y,X_left,y_left,X_right,y_right=get_X_y(filename,session)  
    all_ind=range(len(X))
    scoring,scoring_shuff,scoring_across,scoring_across_shuff=[],[],[],[]
    for i in range(iterations):        
        # Make a random draw from all indices that are not target indices. 
        rand_ind=N.random.choice(all_ind,size=n_neurons,replace=False)
        X_target=X[rand_ind]
        y_shuff=N.random.permutation(y)
        
        pred=cross_val_predict(clf,X_target.T,y,cv=kf) 
        pred_shuff=cross_val_predict(clf,X_target.T,y_shuff,cv=kf)
        scoring_local=mean_absolute_error(y,pred)
        scoring_shuff_local=mean_absolute_error(y,pred_shuff)
        
        scoring.append(N.mean(scoring_local))
        scoring_shuff.append(N.mean(scoring_shuff_local))
        
    scoring=N.mean(scoring)
    scoring_shuff=N.mean(scoring_shuff)
    cm=confusion_matrix(y,pred,normalize='true')
    cm_shuff=confusion_matrix(y,pred_shuff,normalize='true')
    
    cm_total.append(cm)
    cm_shuff_total.append(cm_shuff)
                 
    score_total[animal]=scoring
    score_shuff_total[animal]=scoring_shuff
       
res={'score':score_total,'score shuff':score_shuff_total,'cm':N.asarray(cm_total),'cm shuff':N.asarray(cm_shuff_total),
     'mouse list':animal_list}
os.chdir(target_dir)

N.save("decoding_elapsed_time_%s_start.npy" %start_point,res)

