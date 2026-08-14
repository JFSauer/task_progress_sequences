#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:34:28 2023

@author: jonas
"""

import numpy as N
import os
import h5py
from rastermap import Rastermap
import matplotlib.pyplot as pl

target_dir="/Users/jonas/Documents/projects/rest_move_sequences/results_files" # Directory in which the results file is saved.
data_dir="/Users/jonas/Documents/projects/rest_move_sequences/data/hdf5" # Adjust to data location

N.random.seed(42)

# Executed part.
minima_search_range=10 # Max frames to detect a local minimum following sampling zone entry = true sampling start.

n_PCs=200
n_clusters=100
locality=1
time_lag_window=20

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

trial_starts_left,trial_starts_right=[],[]
trial_ends_left,trial_ends_right=[],[]

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
    rewl=N.asarray(f['%s/behavior/rewl'%session])
    rewr=N.asarray(f['%s/behavior/rewr'%session])
        
    # Get task-phase start points start_left and start_right and end points as phase_end_left and phase_end_right for left and right (i.e., nose poke into odour port to rewar plus reward_offset).
    y_raw=track[1]
    phase_start_left=N.empty((len(sampl[0::2])),dtype="int")
    phase_start_right=N.empty((len(sampr[0::2])),dtype="int")
    phase_end_left=N.empty((len(sampl[0::2])),dtype="int")
    phase_end_right=N.empty((len(sampr[0::2])),dtype="int")

    for n in range(len(sampl[0::2])):
        local=sampl[0::2][n]
        offset=N.argmin(y_raw[local:local+minima_search_range])
        phase_start_left[n]=local+offset
        local=rewl[0::2][n]
        offset=N.argmin(y_raw[local:local+minima_search_range])
        phase_end_left[n]=local+offset
        
    for n in range(len(sampr[0::2])):
        local=sampr[0::2][n]
        offset=N.argmin(y_raw[local:local+minima_search_range])
        phase_start_right[n]=local+offset
        local=rewr[0::2][n]
        offset=N.argmin(y_raw[local:local+minima_search_range])
        phase_end_right[n]=local+offset
        
    trial_duration_left,trial_duration_right=[],[]
    
    trial_starts_left.append(phase_start_left)
    trial_starts_right.append(phase_start_right)
    trial_ends_left.append(phase_end_left)
    trial_ends_right.append(phase_end_right)
        
    ### Fit rastermap model
    # fit rastermap
    print("fitting model")
    model = Rastermap(n_PCs=n_PCs, n_clusters=n_clusters, 
                      locality=locality, time_lag_window=time_lag_window).fit(spks)
    y = model.embedding # neurons x 1
    isort = model.isort
    
    # visualize binning over neurons
    X_embedding = model.X_embedding
    
    res={'trial starts left':sampl[0::2],'trial starts right':sampr[0::2],
         'trial ends left':rewl[0::2],'trial ends right':rewr[0::2],
         'track':track,'spks':spks,'isort':isort}
    
    os.chdir(target_dir)
    N.save('trastermap_analysis_%s.npy' %animal_list[animal],res)


    f,(ax1,ax2)=pl.subplots(2,1,sharex=True)
    ax1.plot(track[0])
    ax1.plot(track[1])
    ax2.pcolormesh(spks[isort])
    ax2.set_xlim(0,5000)
    ax1.set_title("animal: %s, locality: %s, time lag: %s" %(animal_list[animal],locality, time_lag_window))






