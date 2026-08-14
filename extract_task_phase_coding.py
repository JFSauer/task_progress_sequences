#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:34:28 2023

@author: jonas
"""
import numpy as N
import task_progress_sequences_utils as uti
import scipy.stats as st
import os
import h5py
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

target_dir="/Users/jonas/Documents/projects/rest_move_sequences/results_files" # Directory in which the results file is saved.
data_dir="/Users/jonas/Documents/projects/rest_move_sequences/data/hdf5" # Adjust to data location

N.random.seed(42)

def get_phase_tuning(spks,track,start,end,bins=20):

    binned_activity=N.empty((len(spks),len(start),bins))  
    binned_x=N.empty((len(start),bins))
    binned_y=N.empty_like(binned_x)

    for trial in range(len(start)):
        local_activity=spks[:,start[trial]:end[trial]] # Activity of all neurons during the current trial.
        local_x=N.linspace(0,1,end[trial]-start[trial]) # Local phase range from 0 to 1.
        binned,_,_=st.binned_statistic(local_x,local_activity,statistic="mean",bins=bins)
        binned_activity[:,trial,:]=binned
        
        local_track=track[0][start[trial]:end[trial]]
        binned,_,_=st.binned_statistic(local_x,local_track,statistic="mean",bins=bins)
        binned_x[trial]=binned
        local_track=track[1][start[trial]:end[trial]]
        binned,_,_=st.binned_statistic(local_x,local_track,statistic="mean",bins=bins)
        binned_y[trial]=binned
    
    binned_activity_av=N.nanmean(binned_activity,axis=1)
        
    return binned_activity,binned_activity_av,binned_x,binned_y
  
    
def get_correlation_left_right(left,right):
    rand_ind=N.random.permutation(range(len(right)))
    data_shuff=right[rand_ind]
    
    c,c_shuff=[],[]
    for n in range(len(left)):
        c_local,_=st.pearsonr(left[n],right[n])
        c.append(c_local)
        c_local,_=st.pearsonr(left[n],data_shuff[n])
        c_shuff.append(c_local)
        
    return c,c_shuff

def get_consistency(data):
    # Separate the data in first v. second half:
        data_first=N.nanmean(data[:,:int(len(data[1])/2),:],axis=1) # First half
        data_second=N.nanmean(data[:,int(len(data[1])/2):,:],axis=1) # Second half
        rand_ind=N.random.permutation(range(len(data_second)))
        data_shuff=data_second[rand_ind]
        c,c_shuff=[],[]
        for n in range(len(data_first)):
            c_local,_=st.pearsonr(data_first[n],data_second[n])
            c.append(c_local)
            c_local,_=st.pearsonr(data_first[n],data_shuff[n])
            c_shuff.append(c_local)
            
        return c,c_shuff,data_first,data_second
    
def get_Xy_for_decoding(binned_activity):
    n_trials=len(binned_activity[0])
    n_neurons=len(binned_activity)
    X=N.reshape(binned_activity,[n_neurons,-1]) # Resort the array as neurons x trials.
    y=N.tile(range(bins),n_trials)
    
    return X.T,y
    

# Executed part.
minima_search_range=10 # Max frames to detect a local minimum following sampling zone entry = true sampling start.
bins=20 # Number of bins used for phase binning.
sampling_bins=5 # Number of bins used for phase binning during the sampling phase.
reward_offset=10 # Number of sampling points used as phase end point after reward onset.
n_neurons=100 # Number of neurons used for subsampling (decoding).
iterations=100 # Number of iterations for decoding.


params={'minima search range':minima_search_range,
        'bins':bins,
        'sampling bins':sampling_bins}

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
# Inclusion criteria: At least 100 neurons recorded. 47 and 93 are therefore excluded.

binned_phase_left_total,binned_phase_right_total=[],[]
binned_phase_left_av_total,binned_phase_right_av_total=[],[]
binned_space_left_av_total,binned_space_right_av_total=[],[]

phase_correlation_left_right_total,phase_correlation_left_right_by_mouse=[],[]
phase_correlation_left_right_shuff_by_mouse=[]

phase_correlation_sampling_left_right_by_mouse=[]
phase_correlation_sampling_left_right_shuff_by_mouse=[]

phase_consistency_by_mouse,phase_consistency_total=[],[]
phase_consistency_shuff_by_mouse=[]

phase_consistency_sampling_by_mouse=[]
phase_consistency_sampling_shuff_by_mouse=[]

binned_x_left_total, binned_y_left_total = [],[]
binned_x_right_total, binned_y_right_total = [],[]

score_rand_total,score_total=[],[]
score_across_rand_total,score_across_total=[],[]

y_pred_left_total,y_pred_right_total=[],[]
y_pred_across_left_total,y_pred_across_right_total=[],[]
y_left_total,y_right_total=[],[]

y_pred_across_left_binned_total,y_pred_across_right_binned_total=[],[]
y_pred_left_binned_total,y_pred_right_binned_total=[],[]

params['animal list']=animal_list
counter=0
for animal in range(len(animal_list)):
    print(animal_list[animal])

    session=session_list[animal]
    filename=filename_list[animal]    
    
    f = h5py.File(filename, 'r')
    
    spks=N.asarray(f['%s/analysis/trace/Fn' %session])
    track=N.asarray(f['%s/behavior/track'%session])
    corr=N.asarray(f['%s/behavior/corr'%session])  
    sampl=N.asarray(f['%s/behavior/sampl'%session])
    sampr=N.asarray(f['%s/behavior/sampr'%session])
    ocl=N.asarray(f['%s/behavior/ocl'%session])
    ocr=N.asarray(f['%s/behavior/ocr'%session])
    osl=N.asarray(f['%s/behavior/osl'%session])
    osr=N.asarray(f['%s/behavior/osr'%session])
    rewl=N.asarray(f['%s/behavior/rewl'%session])
    rewr=N.asarray(f['%s/behavior/rewr'%session])
    
    # Restrict all periods to correct trials.
    sampl=uti.separate_in(sampl,corr)
    sampr=uti.separate_in(sampr,corr)
    ocl=uti.separate_in(ocl,corr)
    ocr=uti.separate_in(ocr,corr)
    osl=uti.separate_in(osl,corr)
    osr=uti.separate_in(osr,corr)
    rewl=uti.separate_in(rewl,corr)
    rewr=uti.separate_in(rewr,corr)
    
  
    
    # Get task-phase start points start_left and start_right and end points as phase_end_left and phase_end_right for left and right (i.e., nose poke into odour port to reward plus reward_offset).
    y_raw=track[1]
    phase_start_left=N.empty((len(sampl[0::2])),dtype="int")
    phase_start_right=N.empty((len(sampr[0::2])),dtype="int")
    phase_end_left=N.empty((len(sampl[0::2])),dtype="int")
    phase_end_right=N.empty((len(sampr[0::2])),dtype="int")
    phase_end_sampling_left=sampl[1::2]
    phase_end_sampling_right=sampr[1::2]
    for n in range(len(sampl[0::2])):
        local=sampl[0::2][n]
        offset=N.argmin(y_raw[local:local+minima_search_range])
        phase_start_left[n]=local+offset
        phase_end_left[n]=rewl[0::2][n]+reward_offset
    for n in range(len(sampr[0::2])):
        local=sampr[0::2][n]
        offset=N.argmin(y_raw[local:local+minima_search_range])
        phase_start_right[n]=local+offset
        phase_end_right[n]=rewr[0::2][n]+reward_offset
        
        
    binned_phase_left,binned_phase_left_av,binned_x_left,binned_y_left=get_phase_tuning(spks,track,phase_start_left,phase_end_left,bins=bins)
    binned_phase_right,binned_phase_right_av,binned_x_right,binned_y_right=get_phase_tuning(spks,track,phase_start_right,phase_end_right,bins=bins)
    if counter==0:
        binned_phase_left_av_total=binned_phase_left_av
        binned_phase_right_av_total=binned_phase_right_av
    else:
        binned_phase_left_av_total=N.append(binned_phase_left_av_total,binned_phase_left_av,axis=0)
        binned_phase_right_av_total=N.append(binned_phase_right_av_total,binned_phase_right_av,axis=0)
        
    binned_x_left_total.append(binned_x_left)
    binned_x_right_total.append(binned_x_right)
    binned_y_left_total.append(binned_y_left)
    binned_y_right_total.append(binned_y_right)
    
    binned_phase_left_total.append(binned_phase_left)
    binned_phase_right_total.append(binned_phase_right)
        
    phase_correlation_left_right,phase_correlation_left_right_shuff=get_correlation_left_right(binned_phase_left_av,binned_phase_right_av)
    
    phase_correlation_left_right_by_mouse.append(N.nanmean(phase_correlation_left_right))
    phase_correlation_left_right_shuff_by_mouse.append(N.nanmean(phase_correlation_left_right_shuff))
    phase_correlation_left_right_total.extend(phase_correlation_left_right)
    
    
    # Determine within-trial type consistency:
    phase_consistency_left,phase_consistency_left_shuff,binned_phase_first_left,binned_phase_second_left=get_consistency(binned_phase_left)
    phase_consistency_right,phase_consistency_right_shuff,binned_phase_first_right,binned_phase_second_right=get_consistency(binned_phase_right)
    
    if counter==0:
        binned_phase_first_left_total=binned_phase_first_left
        binned_phase_second_left_total=binned_phase_second_left
    else:
        binned_phase_first_left_total=N.append(binned_phase_first_left_total,binned_phase_first_left,axis=0)
        binned_phase_second_left_total=N.append(binned_phase_second_left_total,binned_phase_second_left,axis=0)
    
    
    phase_consistency_average=N.empty((2,len(phase_consistency_left)))
    phase_consistency_average[0]=phase_consistency_left
    phase_consistency_average[1]=phase_consistency_right
    phase_consistency_average=N.nanmean(phase_consistency_average,axis=0)
    
    phase_consistency_average_shuff=N.empty((2,len(phase_consistency_left)))
    phase_consistency_average_shuff[0]=phase_consistency_left_shuff
    phase_consistency_average_shuff[1]=phase_consistency_right_shuff
    phase_consistency_average_shuff=N.nanmean(phase_consistency_average_shuff,axis=0)
    
    phase_consistency_total.extend(phase_consistency_average)
    phase_consistency_by_mouse.append(N.nanmean(phase_consistency_average))
    phase_consistency_shuff_by_mouse.append(N.nanmean(phase_consistency_average_shuff))
    
    
    binned_phase_sampling_left,binned_phase_sampling_left_av,_,_=get_phase_tuning(spks,track,phase_start_left,phase_end_sampling_left,bins=sampling_bins)
    binned_phase_sampling_right,binned_phase_sampling_right_av,_,_=get_phase_tuning(spks,track,phase_start_right,phase_end_sampling_right,bins=sampling_bins)
            
    phase_correlation_sampling_left_right,phase_correlation_sampling_left_right_shuff=get_correlation_left_right(binned_phase_sampling_left_av,binned_phase_sampling_right_av)
    
    phase_correlation_sampling_left_right_by_mouse.append(N.nanmean(phase_correlation_sampling_left_right))
    phase_correlation_sampling_left_right_shuff_by_mouse.append(N.nanmean(phase_correlation_sampling_left_right_shuff))

    
    
    # Determine within-trial type consistency:
    phase_consistency_sampling_left,phase_consistency_sampling_left_shuff,_,_=get_consistency(binned_phase_sampling_left)
    phase_consistency_sampling_right,phase_consistency_sampling_right_shuff,_,_=get_consistency(binned_phase_sampling_right)
    
    
    phase_consistency_sampling_average=N.empty((2,len(phase_consistency_sampling_left)))
    phase_consistency_sampling_average[0]=phase_consistency_sampling_left
    phase_consistency_sampling_average[1]=phase_consistency_sampling_right
    phase_consistency_sampling_average=N.nanmean(phase_consistency_sampling_average,axis=0)
    
    phase_consistency_sampling_average_shuff=N.empty((2,len(phase_consistency_sampling_left)))
    phase_consistency_sampling_average_shuff[0]=phase_consistency_sampling_left_shuff
    phase_consistency_sampling_average_shuff[1]=phase_consistency_sampling_right_shuff
    phase_consistency_sampling_average_shuff=N.nanmean(phase_consistency_sampling_average_shuff,axis=0)
    

    phase_consistency_sampling_by_mouse.append(N.nanmean(phase_consistency_sampling_average))
    phase_consistency_sampling_shuff_by_mouse.append(N.nanmean(phase_consistency_sampling_average_shuff))
    
    
    
    ### Decode trial phase from neuronal activity.

    kf=KFold(n_splits=5,shuffle=True)
    model=LinearRegression(tol=1e-8)
       
    X_total,y=get_Xy_for_decoding(binned_phase_left)
    
    score_left,score_left_rand=[],[]    
    y_pred_left=N.empty((iterations,len(y)))
    y_pred_left_binned=N.empty((iterations,bins))
        
    for i in range(iterations):
        rand_ind=N.random.choice(len(X_total[0]),size=n_neurons,replace=False)
        X=X_total[:,rand_ind]
        
        y_pred=cross_val_predict(model,X,y,cv=kf)
        binned,_,_=st.binned_statistic(y,y_pred,bins=bins,statistic="mean")
        score_left.append(mean_absolute_error(y,y_pred))
        y_rand=N.random.permutation(y)
        score_left_rand.append(mean_absolute_error(y_rand,y_pred))
        
        y_pred_left[i]=y_pred
        y_pred_left_binned[i]=binned
        
    score_left=N.mean(score_left)    
    y_pred_left_total.append(N.mean(y_pred_left,axis=0))
    y_left_total.append(y)
    y_pred_left_binned_total.append(N.mean(y_pred_left_binned,axis=0))
            
    X_total,y=get_Xy_for_decoding(binned_phase_right)
    score_right,score_right_rand=[],[]
    y_pred_right=N.empty((iterations,len(y)))
    y_pred_right_binned=N.empty((iterations,bins))
    
    for i in range(iterations):
        rand_ind=N.random.choice(len(X_total[0]),size=n_neurons,replace=False)
        X=X_total[:,rand_ind]
        
        y_pred=cross_val_predict(model,X,y,cv=kf)
        binned,_,_=st.binned_statistic(y,y_pred,bins=bins,statistic="mean")
        score_right.append(mean_absolute_error(y,y_pred))
        y_rand=N.random.permutation(y)
        score_right_rand.append(mean_absolute_error(y_rand,y_pred))
        
        y_pred_right[i]=y_pred
        y_pred_right_binned[i]=binned
    
    score_right=N.mean(score_right)
    y_pred_right_total.append(N.mean(y_pred_right,axis=0))
    y_right_total.append(y)
    y_pred_right_binned_total.append(N.mean(y_pred_right_binned,axis=0))
    
    
    score=N.mean([score_left,score_right])
    score_rand=N.mean([score_left_rand,score_right_rand])
    
    score_total.append(score/bins) # Normalize error to bin number to get values between 0 and 1.
    score_rand_total.append(score_rand/bins)
    
    
    ### Decode across trial types from neuronal activity.
    
    X_left_total,y_left=get_Xy_for_decoding(binned_phase_left)
    X_right_total,y_right=get_Xy_for_decoding(binned_phase_right)
    
    
    def decode_across(X_train_total,X_test_total,y_train,y_test):
        
        score,score_rand=[],[]
        y_pred=N.empty((iterations,len(y_test)))
        y_pred_binned=N.empty((iterations,bins))
        for i in range(iterations):
            rand_ind=N.random.choice(len(X_train_total[0]),size=n_neurons,replace=False)
            X_train=X_train_total[:,rand_ind]
            X_test=X_test_total[:,rand_ind]
            
            trained=model.fit(X_train,y_train)
            pred=trained.predict(X_test)
            score.append(mean_absolute_error(y_test,pred))
            y_rand=N.random.permutation(y_test)
            score_rand.append(mean_absolute_error(y_rand,pred))
            
                        
            binned,_,_=st.binned_statistic(y_test,pred,bins=bins,statistic="mean")                                                
            y_pred[i]=pred
            y_pred_binned[i]=binned
            
        return N.mean(score), N.mean(score_rand), y_pred, y_pred_binned
    
    
    score_across1, score_across_rand1,y_pred1, y_pred_binned1 = decode_across(X_left_total,X_right_total,y_left,y_right)
    score_across2, score_across_rand2,y_pred2, y_pred_binned2 = decode_across(X_right_total,X_left_total,y_right,y_left)
    
    score_across=N.mean([score_across1,score_across2])
    score_across_total.append(score_across/bins) # Normalize error to bin number to get values between 0 and 1.
    
    score_across_rand=N.mean([score_across_rand1,score_across_rand2])
    score_across_rand_total.append(score_across_rand/bins) # Normalize error to bin number to get values between 0 and 1.
    
    
    y_pred_across_left_binned_total.append(N.mean(y_pred_binned1,axis=0))
    y_pred_across_right_binned_total.append(N.mean(y_pred_binned2,axis=0))
    
    y_pred_across_left_total.append(N.mean(y_pred1,axis=0))
    y_pred_across_right_total.append(N.mean(y_pred2,axis=0))
        
    counter+=1

for_decoding={'binned x left':binned_x_left_total,'binned y left':binned_y_left_total,'binned x right':binned_x_right_total,'binned y right':binned_y_right_total,
              'binned phase left':binned_phase_left_total,'binned phase right':binned_phase_right_total,
              'score':N.asarray(score_total),'score rand':N.asarray(score_rand_total),
              'score across':N.asarray(score_across_total),'score across rand':N.asarray(score_across_rand_total),
              'y left':y_left_total,'y right':y_right_total,
              'y_pred left':y_pred_left_total,'y_pred right':y_pred_right_total,
              'y_pred across left':y_pred_across_left_total,'y_pred across right':y_pred_across_right_total,
              'y_pred binned left':N.asarray(y_pred_left_binned_total),'y_pred binned right':N.asarray(y_pred_right_binned_total),
              'y_pred across binned left':N.asarray(y_pred_across_left_binned_total),'y_pred across binned right':N.asarray(y_pred_across_right_binned_total)}

res={'phase correlation left right':N.asarray(phase_correlation_left_right_total),
     'phase correlation left right by mouse':N.asarray(phase_correlation_left_right_by_mouse),'phase correlation left right shuff by mouse':N.asarray(phase_correlation_left_right_shuff_by_mouse),
     'phase correlation sampling left right by mouse':N.asarray(phase_correlation_sampling_left_right_by_mouse),'phase correlation sampling left right shuff by mouse':N.asarray(phase_correlation_sampling_left_right_shuff_by_mouse),
              'phase left':binned_phase_left_av_total,'phase right':binned_phase_right_av_total,
              'phase left first':binned_phase_first_left_total,'phase left second':binned_phase_second_left_total,
              'phase consistency by mouse':N.asarray(phase_consistency_by_mouse),'phase consistency total':N.asarray(phase_consistency_total),
              'phase consistency shuff by mouse':N.asarray(phase_consistency_shuff_by_mouse),
              'phase consistency sampling by mouse':N.asarray(phase_consistency_sampling_by_mouse),
              'phase consistency sampling shuff by mouse':N.asarray(phase_consistency_sampling_shuff_by_mouse),'for decoding':for_decoding}




os.chdir(target_dir)
N.save('task_phase_detection_results.npy',res)

