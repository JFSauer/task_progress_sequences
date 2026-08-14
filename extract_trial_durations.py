#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:34:28 2023

@author: jonas
"""
import numpy as N
import task_progress_sequences_utils as uti
import os
import h5py

target_dir="/Users/jonas/Documents/projects/rest_move_sequences/results_files" # Directory in which the results file is saved.
data_dir="/Users/jonas/Documents/projects/rest_move_sequences/data/hdf5" # Adjust to data location

N.random.seed(42)

# Executed part.
minima_search_range=10 # Max frames to detect a local minimum following sampling zone entry = true sampling start.
Fs=20.0 # Sampling frequency of the recordin.
sampling_bins=5 # Number of bins used for phase binning during the sampling phase.


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

trial_duration_left_total,trial_duration_right_total=[],[]
trial_duration_left_by_mouse,trial_duration_right_by_mouse=[],[]
performance=[]
n_cells=[]

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
    
    # Measure performance.   
    n_trials=len(sampl[0::2])+len(sampr[0::2])
    n_corr=len(corr[0::2])
    performance.append(n_corr/n_trials)
    
    # Measure cell number.
    n_cells.append(len(spks))
    
    # Restrict all periods to correct trials.
    sampl=uti.separate_in(sampl,corr)
    sampr=uti.separate_in(sampr,corr)
    rewl=uti.separate_in(rewl,corr)
    rewr=uti.separate_in(rewr,corr)
        
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
    
    for n in range(len(phase_start_left)):
        trial_duration_left.append((phase_end_left[n]-phase_start_left[n])/Fs)
    for n in range(len(phase_start_right)):   
        trial_duration_right.append((phase_end_right[n]-phase_start_right[n])/Fs)
        
    
    trial_duration_left_total.extend(trial_duration_left)
    trial_duration_right_total.extend(trial_duration_right)
    trial_duration_left_by_mouse.append(N.mean(trial_duration_left))
    trial_duration_right_by_mouse.append(N.mean(trial_duration_right))
    
    
    
    
res={'trial duration left total':N.asarray(trial_duration_left_total),'trial duration right total':N.asarray(trial_duration_right_total),
     'trial duration left by mouse':N.asarray(trial_duration_left_by_mouse),'trial duration right by mouse':N.asarray(trial_duration_right_by_mouse),
     'performance':performance,'n cells':n_cells}




os.chdir(target_dir)
N.save('trial_duration_analysis.npy',res)

