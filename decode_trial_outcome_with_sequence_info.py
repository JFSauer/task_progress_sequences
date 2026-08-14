#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 12:39:01 2023

@author: jonas
"""

import h5py
import numpy as N
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

target_dir="/Users/jonas/Documents/projects/rest_move_sequences/results_files" # Directory in which the results file is saved.
data_dir="/Users/jonas/Documents/projects/rest_move_sequences/data/hdf5" # Adjust to data location

# Params
minima_search_range=10 # Max frames to detect a local minimum following sampling zone entry = true sampling start.
n_frames_list=[20,40,80]
n_neurons=100

start_point='samp' # Which variable to use. Options are 'samp', 'oc', 'os', 'rew'

n_splits=5
C=3
max_iter=1000000000
tol=0.000000000001
iterations=100
roll_range=range(40,400,1) # Range of time-shift for the signal in X_shuff (in dp). The time shift is performed once on the raw data.


# Model.
clf=LogisticRegression(C=C,max_iter=max_iter,tol=tol)
kf=KFold(n_splits=n_splits,shuffle=True)

params={'n_frames':n_frames_list,
        'n neurons':n_neurons,
        'start point':start_point,
        'n_splits':n_splits,
        'C':C,
        'max_iter':max_iter,
        'tol':tol,
        'iterations':iterations}

# Seeding.
N.random.seed(42)

# Helper functions.
def get_spatial_map_peak_locations(res):
    ''' Applied to the output of spatial histogram of outward data.
    '''
    peak=[]
    for n in range(len(res)):
        peak.append(N.argmax(res[n]))
    
    return N.asarray(peak)
def separate_in_2d_array(data,manual_in):

    start=manual_in[0::2]
    end=manual_in[1::2]
    
    extracted=[]
    for i in range(len(data)):
        for n in range(len(start)):
            extracted.extend(data[i][int(start[n]):int(end[n])])
    extracted=N.reshape(extracted,[len(data),-1])
    return extracted

def separate_in_1d_array(data,manual_in):

    start=manual_in[0::2]
    end=manual_in[1::2]
    
    extracted=[]
    for n in range(len(start)):
        extracted.extend(data[int(start[n]):int(end[n])])
    
    return N.asarray(extracted)

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
    
    
    ranks,ranks_shifted,y=[],[],[]
    mean_act=[]
    
    # Extract temporal rank and mean activity of each neuron.
    for n in range(len(sampl)):
        start=sampl[n]
        end=sampl[n]+n_frames
        
        ind=get_spatial_map_peak_locations(spks[:,start:end])
        ranks.extend(ind)
        y.append(0) # Zeros for left trials.
        
        mean_act.extend(N.nanmean(spks[:,start:end],axis=1)) # Get the temporal average activity of all neurons.
    
    for n in range(len(sampr)):
        start=sampr[n]
        end=sampr[n]+n_frames
        
        ind=get_spatial_map_peak_locations(spks[:,start:end])
        ranks.extend(ind)
        y.append(1) # Ones for right trials.
        
        mean_act.extend(N.nanmean(spks[:,start:end],axis=1)) # Get the temporal average activity of all neurons.
        
    ranks=N.reshape(ranks,[len(y),-1]).T
    ranks_shifted=N.reshape(ranks_shifted,[len(y),-1]).T
    mean_act=N.reshape(mean_act,[len(y),-1]).T
    
    return N.asarray(y),ranks,mean_act

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
    
score_total=N.empty((len(animal_list),len(n_frames_list)))
score_shuff_total=N.empty_like(score_total)
score_rate_total=N.empty_like(score_total)

for animal in range(len(animal_list)):
    filename=filename_list[animal]
    name=animal_list[animal]
    session=session_list[animal]
    print(name)
        
    for k in range(len(n_frames_list)):
        n_frames=n_frames_list[k]
        y,X,X_rate=get_X_y(filename,session)  
        all_ind=range(len(X))
        scoring,scoring_shuff,scoring_rate=[],[],[]
        for i in range(iterations):        
            # Make a random draw from all indices that are not target indices. 
            rand_ind=N.random.choice(all_ind,size=n_neurons,replace=False)
            X_target=X[rand_ind]
            scoring.append(N.mean(cross_val_score(clf,X_target.T,y,cv=kf)))
            y_shuff=N.random.permutation(y)
            scoring_shuff.append(N.mean(cross_val_score(clf,X_target.T,y_shuff,cv=kf)))
            #pred=cross_val_predict(clf,X_target.T,y,cv=kf) 
            #pred_shuff=cross_val_predict(clf,X_target.T,N.random.permutation(y),cv=kf)
            #scoring.append(accuracy_score(y,pred))
            #scoring_shuff.append(accuracy_score(y,pred_shuff))
            
            X_target=X_rate[rand_ind]

            scoring_rate.append(N.mean(cross_val_score(clf,X_target.T,y,cv=kf)))
            #scoring_rate.append(accuracy_score(y,pred))
        scoring=N.mean(scoring)
        scoring_shuff=N.mean(scoring_shuff)
        scoring_rate=N.mean(scoring_rate)
                 
        score_total[animal][k]=scoring
        score_shuff_total[animal][k]=scoring_shuff
        score_rate_total[animal][k]=scoring_rate
    
res={'score':score_total,'score shuff':score_shuff_total,'score rate':score_rate_total,
     'params':params,'mouse list':animal_list}
os.chdir(target_dir)

N.save("decoding_trial_outcome_%s_start.npy" %start_point,res)

