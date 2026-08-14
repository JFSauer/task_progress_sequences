#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:34:28 2023

@author: jonas
"""
from sklearn.metrics import mean_absolute_error as mae
import numpy as N
import task_progress_sequences_utils as uti
import scipy.signal as si
import scipy.stats as st
import os
import h5py

target_dir="/Users/jonas/Documents/projects/rest_move_sequences/results_files" # Directory in which the results file is saved.
data_dir="/Users/jonas/Documents/projects/rest_move_sequences/data/hdf5" # Adjust to data location

N.random.seed(42)

def convolve_spike_train_with_gaussian(raw,su,sd):
    train=create_spike_train(su,raw)
    gauss=si.windows.gaussian(len(train),sd)
    strain=si.fftconvolve(train,gauss,mode="same")
    return st.zscore(strain)  

def create_spike_train(su,raw):
    train=N.zeros((len(raw)))
    ind=[]
    for n in range(len(su)):
        ind.append(int(su[n]))
    
    for n in ind:
        if n<len(train):
            train[n]=1
    return train

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

def find_sequence(data,seq_len_test,iterations,n_trials,rel_pos,left_right_list,step_size=1,random=False):
    
    time_first=[0,seq_len_test]
    first=data[:,time_first[0]:time_first[1]]
    ind_first=uti.get_spatial_map_peak_locations(first) # Get the peak locations of the template.

    # Get mae to a randomly shuffled order as comparison. Done iteration times.
    shuff=[]
    for i in range(iterations):
        first_shuff=N.empty_like(first)
        for n in range(len(first_shuff)):
            first_shuff[n]=N.roll(first[n],N.random.choice(roll_range))
        ind_shuff=uti.get_spatial_map_peak_locations(first_shuff)
        shuff.append(mae(ind_first,ind_shuff)/len(first))

    sig=[]
    
    for n in range(seq_len_full,len(data[0])-seq_len_test,step_size):
        time_second=[n,n+seq_len_test]
        ind_second=uti.get_spatial_map_peak_locations(data[:,time_second[0]:time_second[1]]) # Get the peak locations at this time step.
        mae_actual=mae(ind_first,ind_second)/len(first) # Get the mae for this time point.
        
        if mae_actual<N.percentile(shuff,stat_thres):
            sig.append(n)
                        
    # Extract the significant peaks by convolving the multiple-detected starting locations.       
    sig_conv=convolve_spike_train_with_gaussian(range(len(data[0])),sig,sd)
    thres=N.std(sig_conv)
    peaks,_=si.find_peaks(sig_conv,height=thres)
    freq=len(peaks)/(n_trials-1)
    freq_total=len(peaks)
    
    # Extract the expression strength for detected peak locations.
    strength=[]
    strength_within,strength_across=[],[]
    
    # Check whether the current template occurred in a left or right trial.
    if left_right_list[0]==0:
        temp_is_left=True
    else:
        temp_is_left=False
       
    for n in peaks:
        time_second=[n,n+seq_len_test]
        ind_second=uti.get_spatial_map_peak_locations(data[:,time_second[0]:time_second[1]])
        r,p=st.spearmanr(ind_first,ind_second) # Get pearson's r between the template and matched peak locations
        strength.append(r)
        
        # Check whether the current match is left or right.
        if left_right_list[n]==0:
            match_is_left=True
        else:
            match_is_left=False
            
        # Sort the strength results.
        if temp_is_left==True:
            if match_is_left==True:
                strength_within.append(r)
            elif match_is_left==False:
                strength_across.append(r)
        elif temp_is_left==False:
            if match_is_left==False:
                strength_within.append(r)
            elif match_is_left==True:
                strength_across.append(r)
            
    
    # Find starting positions of detected sequence.
    starting=rel_pos[peaks]
       
    return peaks,freq,freq_total,N.nanmean(strength),N.nanmean(strength_within),N.nanmean(strength_across),starting
   
def get_data(filename,session):
    f = h5py.File(filename, 'r')
    
    spks_raw=N.asarray(f['%s/analysis/trace/Fn' %session])
    track_raw=N.asarray(f['%s/behavior/track'%session])
        
    # Restrict the data to correct runs and concatenate run data aligned to sampling start. 
    corr=N.asarray(f['%s/behavior/corr'%session])
    ends=corr[1::2]
    # Find true sampling starts from local minima in y coordinate following trial starts.
    starts=N.zeros((len(corr[0::2])),dtype="int")
    y_raw=track_raw[1]
    for n in range(len(corr[0::2])):
        local=corr[0::2][n]
        offset=N.argmin(y_raw[local:local+minima_search_range])
        starts[n]=local+offset
            
    x,y=[],[]
    spks=N.empty((len(spks_raw),0))
    spks_shuff=N.empty_like(spks)
    is_left_right=[]
    rel_pos=[]
    trial_starts=[0]
    trial_start_sums=0
    for n in range(len(corr[0::2])):
        local_len=ends[n]-starts[n]
        temp=N.linspace(0,1,local_len)
        trial_start_sums+=len(temp)
        trial_starts.append(trial_start_sums)
        rel_pos.extend(temp)
        x.extend(track_raw[0][starts[n]:ends[n]])
        y.extend(track_raw[1][starts[n]:ends[n]])
        spks=N.append(spks,spks_raw[:,starts[n]:ends[n]],axis=1)
        spks_raw_shuff=N.empty((len(spks),ends[n]-starts[n]))
        for i in range(len(spks)):
            spks_local=spks_raw[i,starts[n]:ends[n]]
            spks_local=N.roll(spks_local,N.random.choice(roll_range))
            spks_raw_shuff[i]=spks_local
        spks_shuff=N.append(spks_shuff,spks_raw_shuff,axis=1)
        
        # Extract as a continuous array whether we are in a left or right trial.
        local_x=track_raw[0][starts[n]:ends[n]]
        if N.mean(local_x)<0:
            is_left_right.extend(N.zeros((len(local_x)))) # For left
        else:
            is_left_right.extend(N.ones((len(local_x))))
     
    track=N.empty((2,len(x)))
    track[0]=x
    track[1]=y
    
    del(trial_starts[-1])
    
    return spks,track,is_left_right,spks_shuff,N.asarray(trial_starts),N.asarray(rel_pos)

def get_n_trials(filename,session):
    f = h5py.File(filename, 'r')
    corr=N.asarray(f['%s/behavior/corr'%session])
    n_trials=len(corr[0::2])
    
    rel_pos=[]
    
    for n in range(len(corr[0::2])):
        local_len=corr[1::2][n]-corr[0::2][n]
        temp=N.linspace(0,1,local_len)
        rel_pos.extend(temp)
        
    trial_starts=[]
    trial_starts.append(0)
    corr_corrected=N.asarray(corr)-corr[0::2][0] # Let this start at 0
    starts=corr_corrected[0::2]
    ends=corr_corrected[1::2]
    sums=0
    for n in range(len(starts)):
        sums+=ends[n]-starts[n]
        trial_starts.append(sums)
        
    return n_trials,N.asarray(rel_pos),trial_starts

def get_individual_runs(data,peaks,seq_len_full):
        
    data_peaks=[]
    
    for n in range(len(peaks)):
        data_peaks.append(data[:,peaks[n]:peaks[n]+seq_len_full])
             
    return data_peaks
    
def run_analysis(animal_list,session_list,filename_list,is_learning=False):
    detected_start_indices=[]
    react_freq,react_freq_shuff=[],[]
    expr_strength=[]
    expr_strength_mouse_av=[]
    expr_strength_within_mouse_av,expr_strength_across_mouse_av=[],[]
    expr_strength_within_mouse_av_shuff,expr_strength_across_mouse_av_shuff=[],[]
    expr_strength_shuff=[]
    track_total,traces_total=[],[]
    data_peaks_total=[]
    seq_start_total=[]
    trial_starts_total=[]
    speed_r_total,speed_r_total_by_mouse=[],[]
    speed_r_total_shuff,speed_r_total_shuff_by_mouse=[],[]

    for animal in range(len(animal_list)):
        print(animal_list[animal])

        session=session_list[animal]
        filename=filename_list[animal]               
        
        # The analysis takes the start of data and bases the extraction on that sorting.
        # To reduce variability, this is run for each trial (based on trial_starts) and averaged.
        # For that, the data are time-shifted to begin at the next trial.
        
        raw_data,raw_track,raw_left_right_list,raw_data_shuff,trial_starts,rel_pos_raw=get_data(filename,session)
        trial_starts_total.append(trial_starts)
        n_trials=len(trial_starts)

        react_freq_local,react_freq_local_shuff=[],[]
        expr_strength_local=[]
        expr_strength_local_within,expr_strength_local_across=[],[]
        expr_strength_local_shuff=[]
        expr_strength_local_within_shuff,expr_strength_local_across_shuff=[],[]
        
        speed_r_local,speed_r_local_shuff=[],[]
        max_points_temporal_local=N.empty((len(trial_starts),seq_len_full)) # empty array trials * seq len
        current_speed_local=N.empty_like(max_points_temporal_local)
        
        # Get speed from track data (in a.u.)
        x=raw_track[0]
        y=raw_track[1]
        raw_speed=N.empty((len(x)-1))
        for n in range(len(x)-1):
            raw_speed[n]=N.sqrt((x[n+1]-x[n])**2+(y[n+1]-y[n])**2)
        raw_speed=N.append(raw_speed,raw_speed[-1])
               
        for n in range(len(trial_starts)):
            print(n, 'of', len(trial_starts))
            local_data=N.roll(raw_data,-trial_starts[n],axis=1) # Negative value means shifting in forward direction.
            local_data_shuff=N.roll(raw_data_shuff,-trial_starts[n],axis=1)
            speed=N.roll(raw_speed,-trial_starts[n])
            rel_pos=N.roll(rel_pos_raw,-trial_starts[n])
            left_right_list=N.roll(raw_left_right_list,-trial_starts[n])
            
            # Get the initial sorting index.
            test_ind=uti.get_spatial_map_sorting_indices(local_data[:,:seq_len_full])
            sorted_data=local_data[test_ind] # Get the full data in the correct sorting.
            sorted_data_shuff=local_data_shuff[test_ind]
            # Get contributing neurons.
            ind=[]
            for i in range(len(sorted_data)):
                if N.max(sorted_data[i])>N.mean(sorted_data[i])+participation_thres*N.std(sorted_data[i]):
                    ind.append(i)
            ind=N.asarray(ind)
            data=sorted_data[ind] # Restrict the data to the cells active during this pattern.
            data_shuff=sorted_data_shuff[ind]
            
            current_template=data[:,:seq_len_full]
            ## Here we have current_template as the ordered activity map of the current template, resticted to active neurons
            current_speed=speed[:seq_len_full]
            
            max_points=uti.get_spatial_map_peak_locations(current_template)
            # Get activation times in the same sampling as speed.
            max_points_temporal,_=N.histogram(max_points,bins=seq_len_full)
            local_corr,_=st.spearmanr(current_speed,max_points_temporal)
            random_current_speed=N.random.permutation(current_speed)
            local_corr_shuff,_=st.spearmanr(random_current_speed,max_points_temporal)

            max_points_temporal_local[n]=max_points_temporal
            current_speed_local[n]=current_speed
            speed_r_local.append(local_corr)
            speed_r_local_shuff.append(local_corr_shuff)
            
            peaks,freq,freq_total,strength,strength_within,strength_across,seq_start=find_sequence(data,seq_len_test,iterations,n_trials,rel_pos,left_right_list,step_size=step_size)        
                   
            react_freq_local.append(freq)
            expr_strength_local.append(strength)
            expr_strength_local_within.append(strength_within)
            expr_strength_local_across.append(strength_across)
            seq_start_total.extend(seq_start)
            
            if n==0:
                data_peaks=get_individual_runs(data,peaks,seq_len_full)
                data_peaks_total.append(data_peaks)
                
            peaks_shuff,freq,freq_total,strength,strength_within,strength_across,seq_start=find_sequence(data_shuff,seq_len_test,iterations,n_trials,rel_pos,left_right_list,step_size=step_size)        
            react_freq_local_shuff.append(freq)
            expr_strength_local_shuff.append(strength)
            expr_strength_local_within_shuff.append(strength_within)
            expr_strength_local_across_shuff.append(strength_across)
                                            
        # Average the relevant metrics (freq + expr strength) over all templates.
        # Keep the individual results for the sequence start time.
        react_freq.append(N.nanmean(react_freq_local))
        react_freq_shuff.append(N.nanmean(react_freq_local_shuff))
        expr_strength_mouse_av.append(N.nanmean(expr_strength_local))
        expr_strength_within_mouse_av.append(N.nanmean(expr_strength_local_within))
        expr_strength_across_mouse_av.append(N.nanmean(expr_strength_local_across))
        expr_strength_within_mouse_av_shuff.append(N.nanmean(expr_strength_local_within_shuff))
        expr_strength_across_mouse_av_shuff.append(N.nanmean(expr_strength_local_across_shuff))
        expr_strength.extend(expr_strength_local)
        expr_strength_shuff.append(N.nanmean(expr_strength_local_shuff))
        
        if animal==0:
            max_points_temporal_total=max_points_temporal_local
            current_speed_total=current_speed_local
        else:
            max_points_temporal_total=N.append(max_points_temporal_total,max_points_temporal_local,axis=0)
            current_speed_total=N.append(current_speed_total,current_speed_local,axis=0)
        
        speed_r_total.extend(speed_r_local)       
        speed_r_total_by_mouse.append(N.nanmean(speed_r_local))
        speed_r_total_shuff.extend(speed_r_local_shuff)       
        speed_r_total_shuff_by_mouse.append(N.nanmean(speed_r_local_shuff))
        
        # Keep examples for illustration from the last iteration.
        detected_start_indices.append(peaks)
        traces_total.append(raw_data)
        track_total.append(raw_track)
                
    local_res={'detected indices':detected_start_indices,
               'expression strength shuff':N.asarray(expr_strength_shuff),'reactivations per trial':N.asarray(react_freq),'reactivations per trial shuff':N.asarray(react_freq_shuff),
             'expression strength':N.asarray(expr_strength_mouse_av),'expression strength total':N.asarray(expr_strength),
             'expression strength within':N.asarray(expr_strength_within_mouse_av),'expression strength across':N.asarray(expr_strength_across_mouse_av),
             'expression strength within shuff':N.asarray(expr_strength_within_mouse_av_shuff),'expression strength across shuff':N.asarray(expr_strength_across_mouse_av_shuff),
             'traces':traces_total,'track':track_total,'seq start points':seq_start_total,
             'data peaks':data_peaks_total,
             'speed neuron r by mouse':N.asarray(speed_r_total_by_mouse),'speed neuron r':N.asarray(speed_r_total),
             'speed neuron r by mouse shuff':N.asarray(speed_r_total_shuff_by_mouse),'speed neuron r shuff':N.asarray(speed_r_total_shuff),
             'max points temporal':max_points_temporal_total,'current speed total':current_speed_total,                 
             'trial starts total':trial_starts_total,
             'params':params}
    return local_res

# Executed part.

seq_len_full=100
seq_len_test=100
minima_search_range=10 # Max frames to detect a local minimum following sampling zone entry = true sampling start.
iterations=500 # Random iterations to identify significant sequences.
sd=20 # SD for convolution in dp.
Fs=20 # Sampling freqency in Hz.
stat_thres=0.002
roll_range=range(40,400,1) # Range of time-shift for the signal in X_shuff (in dp)
participation_thres=2 # Threshold of activity above mean to be considered active in this sequence.
step_size=3 # Step size (in dp) while iterating over the data to find potential sequence start points.

params={'seq len full':seq_len_full,
        'seq len test':seq_len_test,
        'SD':sd,
        'iterations':iterations,
        'roll range':roll_range,
        'stat thres':stat_thres,
        'minima search range':minima_search_range}

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

params['animal list']=animal_list
res=run_analysis(animal_list,session_list,filename_list)

os.chdir(target_dir)
N.save('sequence_detection_results_%s_%s.npy' %(seq_len_full, seq_len_test),res)

