#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 16:39:30 2026

@author: jonas

Code to generate Figures 1 & 2 as well as Supplementary Figures 1-3 of Tarahi et al.
The code reads results files of the following esparate analysis code:
    detect_significant_sequences.py
    detect_significant_sequences_random_time_points.py
    extract_trial_durations.py
    decode_time_from_sampling_onset.py
    decode_trial_outcome_with_sequence_info.py
    extract_task_phase_coding.py
    rastermap_analysis.py
"""

import numpy as N
import scipy.stats as st
import matplotlib.pyplot as pl
import matplotlib.gridspec as gridspec
import colorcet as cc
import task_progress_sequences_utils as uti
import task_progress_sequences_statistical_functions as stats

N.random.seed(125)

legend_fontsize="small"
title_fontsize=10

# Boxplot_properties
medianprops=dict(c="k")

# Colormap properties.
cmap=cc.cm.fire

# Permutation properties.
permutation_iterations=5000
permutation_thres=99

##### General plotting functions. #####

def boxplot_with_lines(ax,data,xlabel=['data','shuffled'],colours=['green','gray'],ylabel="Reactivation proportion",ymin=0,ymax=1,use_legend=True):
    x1=N.ones(len(data[0]))
    x2=x1+1

    
    x_line=[1,2]
    for n in range(len(data.T)):
        _=ax.plot(x_line,data.T[n],color="k",linewidth=0.5)
    
    box=ax.boxplot(data.T,showfliers=False,widths=0.5,medianprops=dict(c="k"),patch_artist=True,capwidths=0.5)  
    colors=["white","white"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        

    ax.scatter(x1,data[0],s=20,c=colours[0],alpha=0.5,edgecolor="k",label=xlabel[0],zorder=3)
    ax.scatter(x2,data[1],s=20,c=colours[1],alpha=0.5,edgecolor="k",label=xlabel[1],zorder=3)
        
    
    ax.set_ylim(ymin,ymax)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel(ylabel)
    if use_legend==True:
        ax.legend(loc=[0.02,0],frameon=False,markerscale=1,alignment="left", labelspacing=0.1,handlelength=0.1)
        ax.xaxis.set_visible(False)
    else:
        ax.set_xticklabels(xlabel)
    
    
def boxplot3_with_lines(ax,data,xlabel=['same','opposite','same shuffled'],colours=['green','blue','gray'],ylabel="Reactivation proportion",ymin=0,ymax=1):
    
    x1=N.ones(len(data[0]))
    x2=x1+1
    x3=x2+1
    
    x_line=[1,2,3]
    for n in range(len(data.T)):
        _=ax.plot(x_line,data.T[n],color="k",linewidth=0.5)
    
    box=ax.boxplot(data.T,showfliers=False,widths=0.5,medianprops=dict(c="k"),patch_artist=True,capwidths=0.5)  
    colors=["white","white","white"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        

    ax.scatter(x1,data[0],s=20,c=colours[0],alpha=0.5,edgecolor="k",label=xlabel[0],zorder=3)
    ax.scatter(x2,data[1],s=20,c=colours[1],alpha=0.5,edgecolor="k",label=xlabel[1],zorder=3)
    ax.scatter(x3,data[2],s=20,c=colours[2],alpha=0.5,edgecolor="k",label=xlabel[2],zorder=3)

    

    ax.set_ylim(ymin,ymax)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower left",frameon=False,markerscale=1,alignment="left", labelspacing=0.1,handlelength=0.1)
    #ax.set_xticklabels(xlabel)
    ax.xaxis.set_visible(False)
    
def boxplot4_with_lines(ax,data,xlabel=['same','same shuffled','opposite','same shuffled'],colours=['green','gray','blue','gray'],ylabel="Reactivation proportion",ymin=0,ymax=1):
    
    x1=N.ones(len(data[0]))
    x2=x1+1
    x3=x2+1
    x4=x3+1
    
    x_line=[1,2,3,4]
    for n in range(len(data.T)):
        _=ax.plot(x_line,data.T[n],color="k",linewidth=0.5)
    
    box=ax.boxplot(data.T,showfliers=False,widths=0.5,medianprops=dict(c="k"),patch_artist=True,capwidths=0.5)  
    colors=["white","white","white","white"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        

    ax.scatter(x1,data[0],s=20,c=colours[0],alpha=0.5,edgecolor="k",label=xlabel[0],zorder=3)
    ax.scatter(x2,data[1],s=20,c=colours[1],alpha=0.5,edgecolor="k",label=xlabel[1],zorder=3)
    ax.scatter(x3,data[2],s=20,c=colours[2],alpha=0.5,edgecolor="k",label=xlabel[2],zorder=3)
    ax.scatter(x4,data[3],s=20,c=colours[3],alpha=0.5,edgecolor="k",label=xlabel[3],zorder=3)

    

    ax.set_ylim(ymin,ymax)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower left",frameon=False,markerscale=1,alignment="left", labelspacing=0.1,handlelength=0.1)
    #ax.set_xticklabels(xlabel)
    ax.xaxis.set_visible(False)
    
    
def plot_siginficance_symbol_anova(ax,data,stat_res,text_position,comparison,anchor,height=0.1,plot_full=False):
    
    bar_position=N.max(data[anchor])+N.max(data[anchor])*height
    
    text_yoffset=bar_position
    
    if stat_res['Pcorr'][comparison]<0.001:
        ax.text(text_position,text_yoffset,"***",ha="center", va="bottom")
        if plot_full==True:  
            ax.hlines(bar_position,1.1,2.8,"k",linewidth=0.8)
        else:
            ax.hlines(bar_position,text_position-0.4,text_position+0.4,"k",linewidth=0.8)
        
    elif 0.001<stat_res['Pcorr'][comparison]<0.01:
        ax.text(text_position,text_yoffset,"**",ha="center", va="bottom")
        if plot_full==True:  
            ax.hlines(bar_position,1.1,2.8,"k",linewidth=0.8)
        else:
            ax.hlines(bar_position,text_position-0.4,text_position+0.4,"k",linewidth=0.8)
        
    elif 0.01<stat_res['Pcorr'][comparison]<0.05:
        ax.text(text_position,text_yoffset,"*",ha="center", va="bottom")
        if plot_full==True:  
            ax.hlines(bar_position,1.1,2.8,"k",linewidth=0.8)
        else:
            ax.hlines(bar_position,text_position-0.4,text_position+0.4,"k",linewidth=0.8)
            
            
def plot_siginficance_symbol_paired(ax,data,stat_res,height=0.1):
    
    bar_position=N.max(data)+N.max(data)*height
    text_position=1.5
    text_yoffset=bar_position
    
    if stat_res['p']<0.001:
        ax.text(text_position,text_yoffset,"***",ha="center", va="bottom")
        ax.hlines(bar_position,text_position-0.4,text_position+0.4,"k",linewidth=0.8)
        
    elif 0.001<stat_res['p']<0.01:
        ax.text(text_position,text_yoffset,"**",ha="center", va="bottom")
        ax.hlines(bar_position,text_position-0.4,text_position+0.4,"k",linewidth=0.8)
        
    elif 0.01<stat_res['p']<0.05:
       ax.text(text_position,text_yoffset,"*",ha="center", va="bottom")
       ax.hlines(bar_position,text_position-0.4,text_position+0.4,"k",linewidth=0.8)           
    
##### Figure 1 #####

def plot_figure1():
    res=N.load("/Users/jonas/Documents/projects/rest_move_sequences/results_files/sequence_detection_results_100_100.npy",allow_pickle=True).item()
    example_mouse=0
    example_scaling=6000
    

    # Plot colorplot insets.
    # Full neuron example.
    fig=pl.figure(tight_layout=True)
    ax=fig.add_subplot()
    example_data=res['traces'][example_mouse]
    n_neurons=len(example_data)
    idx=uti.get_spatial_map_sorting_indices(example_data[:,:res['params']['seq len full']])

    p=ax.pcolormesh(example_data[idx],cmap=cmap)  
    p.set_clim(0,10)
    ax.set_xlim(0,example_scaling)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    
    x=res['track'][example_mouse][0]
    y=res['track'][example_mouse][1]
    speed=N.empty((len(x)-1))
    for n in range(len(x)-1):
        speed[n]=N.sqrt((x[n+1]-x[n])**2+(y[n+1]-y[n])**2)
    
    # Template example.
    fig=pl.figure(tight_layout=True)
    ax=fig.add_subplot()
    example_scaling=125
    p=ax.pcolormesh(example_data[idx],cmap=cmap)  
    p.set_clim(0,10)
    ax.set_xlim(0,example_scaling)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
        
    # Match example.
    fig=pl.figure(tight_layout=True)
    ax=fig.add_subplot()
    match_trial=17
    start_point=res['trial starts total'][example_mouse][match_trial]
    end_point=start_point+example_scaling
    example_scaling=125
    p=ax.pcolormesh(example_data[idx][:,start_point:end_point],cmap=cmap)  
    p.set_clim(0,10)
    ax.set_xlim(0,example_scaling)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # Colour scale.
    fig=pl.figure(tight_layout=True)
    ax=fig.add_subplot()
    example_scaling=125
    p=ax.pcolormesh(example_data[idx],cmap=cmap)  
    p.set_clim(0,10)
    cbar = fig.colorbar(p)

    ax.set_xlim(0,example_scaling)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)    


    # Plot main figure.
    gs=gridspec.GridSpec(5,5)
    fig=pl.figure(tight_layout=True,figsize=[10,8])
    
    
    ##### Panel B #####
    
    example_scaling=6000
    
    # Plot behaviour
    ax=pl.subplot(gs[1,0:4])
    ax.plot(uti.rescale_pos(res['track'][example_mouse][0]), c="k")
    ax.plot(uti.rescale_pos(res['track'][example_mouse][1]), c="grey")
    ax.set_xlim(0,example_scaling)
    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    
    
    # Plot trial starts
    example_data=res['trial starts total'][example_mouse]
    ax.vlines(example_data,0,1,color="k",linestyles="dashed") 
    ax.set_xlim(0,example_scaling)
    ax.spines[['right', 'top']].set_visible(False)
    
    ax.set_ylabel("Position (au)")
    
    # Plot activities
    ax=pl.subplot(gs[2:4,0:4])
    example_data=res['traces'][example_mouse]
    n_neurons=len(example_data)

    
    # Plot trial starts
    example_data=res['trial starts total'][example_mouse]
    ax.vlines(example_data,0,n_neurons,color="w",linestyles="dashed") 
    ax.set_xlim(0,example_scaling)
    ax.set_ylim(0,n_neurons)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xticks([0,1000,2000,3000,4000,5000,6000])
    ax.set_xticklabels(["0","50","100","150","200","250","300"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neurons")
       
    ### Plot template trial (=trial 1)
    
    
    example_scaling=125 
    
    # Plot template trial
    ax=pl.subplot(gs[4,0:2])
    ax.set_ylim(0,n_neurons)          
    ax1=ax.twinx()
    ax1.plot(uti.rescale_pos(res['track'][example_mouse][0],new_max=1), c="w")
    ax1.plot(uti.rescale_pos(res['track'][example_mouse][1],new_max=1),c="grey")
    ax1.plot(uti.rescale_pos(speed,new_max=1),c="m")
    

    ax.spines['top'].set_visible(False)
    ax1.spines['top'].set_visible(False) 
    ax.set_xlim(0,example_scaling)
    #ax.xaxis.set_visible(False)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neurons")
    ax1.set_ylabel("a.u.")
    ax1.set_ylim(0,1)
    ax.set_xticks([0,20,40,60,80,100,120])
    ax.set_xticklabels([0,1,2,3,4,5,6])
    
    
    
    ### Plot match trial
    
    # Plot behaviour
    ax=pl.subplot(gs[4,2:4])
    ax.set_ylim(0,n_neurons)
    ax1=ax.twinx()
    ax1.plot(uti.rescale_pos(res['track'][example_mouse][0],new_max=1)[start_point:end_point], c="w")
    ax1.plot(uti.rescale_pos(res['track'][example_mouse][1],new_max=1)[start_point:end_point],c="grey")
    ax1.plot(uti.rescale_pos(speed,new_max=1)[start_point:end_point],c="m")
    
    ax.spines['top'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    #ax.xaxis.set_visible(False)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neurons")
    ax1.set_ylabel("a.u.")
    ax1.set_ylim(0,1)
    ax.set_xticks([0,20,40,60,80,100,120])
    ax.set_xticklabels([0,1,2,3,4,5,6])
    
   
    ##### Panel D #####
    
    
    ax=pl.subplot(gs[0:2,4])
    data=N.empty((2,9))
    data[0]=res['reactivations per trial']
    data[1]=res['reactivations per trial shuff']
    boxplot_with_lines(ax,data,ylabel="Reactivation proportion",ymin=-0.2,ymax=1.4)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
    # Statistical testing.
    stat_res=stats.paired_comparison_two_groups(data[0],data[1])
    plot_siginficance_symbol_paired(ax,data,stat_res,height=0.08)
     
    ##### Panel E #####
    
    ax=pl.subplot(gs[2:4,4])
    data=N.empty((2,9))
    data[0]=res['expression strength']
    data[1]=res['expression strength shuff']    
    boxplot_with_lines(ax,data,ylabel="Template-match similarity",ymax=0.5)
    # Statistical testing.
    stat_res=stats.paired_comparison_two_groups(data[0],data[1])
    
    plot_siginficance_symbol_paired(ax,data,stat_res,height=0.08)        
        
    #### Panel F ######
    ax=pl.subplot(gs[4,4],polar=True)
    seq_starts=uti.rescale_pos(N.asarray(res['seq start points']),new_min=0,new_max=2*N.pi) # Convert to degrees
    ax.hist(seq_starts,bins=100,color="k")

def plot_figure2():
    
    
    # Plot colorplot insets.
    res=N.load("/Users/jonas/Documents/projects/rest_move_sequences/results_files/task_phase_detection_results.npy",allow_pickle=True).item()
    
    fig=pl.figure(tight_layout=True)
    ax=fig.add_subplot()
    first=res['phase left']
    second=res['phase right']
    idx=uti.get_spatial_map_sorting_indices(first)
    p=ax.pcolormesh(first[idx],cmap=cmap)  
    p.set_clim(0,10)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel("Neurons")
    ax.set_xlabel("Task phase")
    ax.set_xticklabels([0,0.5,1.0])
    ax.set_yticks([0,500,1000,1500])
    ax.set_title("left")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    fig=pl.figure(tight_layout=True)
    ax=fig.add_subplot()
    p=ax.pcolormesh(second[idx],cmap=cmap)  
    p.set_clim(0,10)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel("Neurons")
    ax.set_xlabel("Task phase")
    ax.set_xticklabels([0,0.5,1.0])
    ax.set_yticks([0,500,1000,1500])
    ax.set_title("right")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    fig=pl.figure(tight_layout=True)
    ax=fig.add_subplot()
    first=res['phase left first']
    second=res['phase left second']
    idx=uti.get_spatial_map_sorting_indices(first)
    p=ax.pcolormesh(first[idx],cmap=cmap)  
    p.set_clim(0,10)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel("Neurons")
    ax.set_xlabel("Task phase")
    ax.set_xticklabels([0,0.5,1.0])
    ax.set_yticks([0,500,1000,1500])
    ax.set_title("left")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    fig=pl.figure(tight_layout=True)
    ax=fig.add_subplot()
    p=ax.pcolormesh(second[idx],cmap=cmap)  
    p.set_clim(0,10)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel("Neurons")
    ax.set_xlabel("Task phase")
    ax.set_xticklabels([0,0.5,1.0])
    ax.set_yticks([0,500,1000,1500])
    ax.set_title("right")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # Plot main figure.
    res=N.load("/Users/jonas/Documents/projects/rest_move_sequences/results_files/sequence_detection_results_100_100.npy",allow_pickle=True).item()
    gs=gridspec.GridSpec(5,6)
    fig=pl.figure(tight_layout=True,figsize=[13,8])
    ### Panel same vs. opposite sequences
    
    ax=pl.subplot(gs[0:2,0])
    data=N.empty((3,9))
    data[0]=res['expression strength within']
    data[1]=res['expression strength across']  
    data[2]=res['expression strength within shuff']
    boxplot3_with_lines(ax,data,ylabel="Template-match similarity",ymax=0.6)
    
    
    # Statistical testing & plotting.
    
    stat_res=stats.one_way_repeated_measures_anova_general_three_groups(data[0],data[1],data[2]) 
    plot_siginficance_symbol_anova(ax,data,stat_res,text_position=1.5,comparison=0,anchor=0)
    plot_siginficance_symbol_anova(ax,data,stat_res,text_position=2.5,comparison=2,anchor=1)
    plot_siginficance_symbol_anova(ax,data,stat_res,text_position=2,comparison=1,anchor=0,height=0.2,plot_full=True)
    
            
    ### Panels correlation in phase tuning left v. right.
    
    res=N.load("/Users/jonas/Documents/projects/rest_move_sequences/results_files/task_phase_detection_results.npy",allow_pickle=True).item()
    
    ax=pl.subplot(gs[0:2,1])
     
    ax.plot([])
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel("Neurons")
    ax.set_xlabel("Task phase")
    ax.set_xticklabels([0,0.5,1.0])
    ax.set_yticks([0,500,1000,1500])
    ax.set_title("left")
    ax.set_xlim(0,1)
    
    
    ax=pl.subplot(gs[0:2,2])
     
    ax.plot([])
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel("Neurons")
    ax.set_xlabel("Task phase")
    ax.set_xticklabels([0,0.5,1.0])
    ax.set_yticks([0,500,1000,1500])
    ax.set_title("right")
    ax.set_xlim(0,1)

    ### Panels phase tuning first v. second
    
    ax=pl.subplot(gs[0:2,3])
    
    ax.plot([])
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel("Neurons")
    ax.set_xlabel("Trial phase")
    ax.set_xticklabels([0,0.5,1.0])
    ax.set_yticks([0,500,1000,1500])
    ax.set_title("first half")
    ax.set_xlim(0,1)
    
    ax=pl.subplot(gs[0:2,4])
    
    ax.plot([])
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylabel("Neurons")
    ax.set_xlabel("Trial phase")
    ax.set_xticklabels([0,0.5,1.0])
    ax.set_yticks([0,500,1000,1500])
    ax.set_title("second half")
    ax.set_xlim(0,1)
    
    
    ### Panels summary plots for consistency and similarity.
    
    ax=pl.subplot(gs[0:2,5])
    
    data=N.empty((3,9))
    data[0]=res['phase correlation left right by mouse']
    data[1]=res['phase consistency by mouse']
    data[2]=res['phase correlation left right shuff by mouse']
    
    boxplot3_with_lines(ax,data,xlabel=['similarity','consistency','shuffled'],ylabel="Phase correlation",ymin=-0.6,ymax=1) 
    
    stat_res=stats.one_way_repeated_measures_anova_general_three_groups(data[0],data[1],data[2]) 
    plot_siginficance_symbol_anova(ax,data,stat_res,text_position=1.5,comparison=0,anchor=1)
    plot_siginficance_symbol_anova(ax,data,stat_res,text_position=2.5,comparison=2,anchor=1)
    plot_siginficance_symbol_anova(ax,data,stat_res,text_position=2,comparison=1,anchor=1,height=0.25,plot_full=True)
    
    ax=pl.subplot(gs[2:4,0])
    
    data=N.empty((3,9))
    data[0]=res['phase correlation sampling left right by mouse']
    data[1]=res['phase consistency sampling by mouse']
    data[2]=res['phase correlation sampling left right shuff by mouse']
    
    boxplot3_with_lines(ax,data,xlabel=['similarity','consistency','shuffled'],ylabel="Phase correlation sampling epoch",ymin=-0.6,ymax=1) 
    
    stat_res=stats.one_way_repeated_measures_anova_general_three_groups(data[0],data[1],data[2]) 
    plot_siginficance_symbol_anova(ax,data,stat_res,text_position=1.5,comparison=0,anchor=1)
    plot_siginficance_symbol_anova(ax,data,stat_res,text_position=2.5,comparison=2,anchor=1)
    plot_siginficance_symbol_anova(ax,data,stat_res,text_position=2,comparison=1,anchor=1,height=0.25,plot_full=True)
            
       
    ### Panel task phase decoding.
    
    # Predictions.
    ax=pl.subplot(gs[2,1])
    ax.plot(res['for decoding']['y_pred right'][0],label="predicted",color="green")
    ax.plot(res['for decoding']['y right'][0],label="true",color="k")
    ax.legend(loc=[0,0],frameon=False,markerscale=1,alignment="left", labelspacing=0.1,handlelength=0.1)
    ax.set_xlim(0,100)
    ax.set_ylim(-15,20)
    ax.set_yticks([0,10,20])
    ax.set_yticklabels([0,0.5,1.0])
    ax.set_ylabel("Trial phase")
    ax.spines[['right', 'top']].set_visible(False)
    ax.xaxis.set_visible(False)
    

    ax=pl.subplot(gs[2,2])
    mean=N.mean(res['for decoding']['y_pred binned right'],axis=0)
    sem=st.sem(res['for decoding']['y_pred binned right'],axis=0)
    
    y1=mean-sem
    y2=mean+sem
    x=range(len(y1))
    ax.fill_between(x,y1,y2,alpha=0.2,color="green")
    ax.plot(mean,linewidth=1,color="k")
    ax.set_xticks([0,10,20])
    ax.set_xticklabels([0,0.5,1.0])
    ax.set_yticks([0,10,20])
    ax.set_yticklabels([0,0.5,1])
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel("True phase")
    ax.set_ylabel("Predicted phase")
    
    
    # Summary of decoding.
    
    ax=pl.subplot(gs[2,3])
    data=N.empty((2,9))
    data[0]=res['for decoding']['score']
    data[1]=res['for decoding']['score rand']
    boxplot_with_lines(ax,data,ylabel="Mean absolute error",ymin=0,ymax=1,use_legend=False)
    #ax.set_yticks([0,0.1,0.2,0.3,0.4])
    
    
    stat_res=stats.paired_comparison_two_groups(data[0],data[1])
    plot_siginficance_symbol_paired(ax,data,stat_res,height=0.3)
    
    
    # Confusion matrices - across.

    # Predictions.
    ax=pl.subplot(gs[3,1])
    ax.plot(res['for decoding']['y_pred across right'][0],label="predicted",color="green")
    ax.plot(res['for decoding']['y right'][0],label="true",color="k")
    ax.legend(loc=[0,0],frameon=False,markerscale=1,alignment="left", labelspacing=0.1,handlelength=0.1)
    ax.set_xlim(0,100)
    ax.set_ylim(-15,20)
    ax.set_yticks([0,10,20])
    ax.set_yticklabels([0,0.5,1.0])
    ax.set_ylabel("Trial phase")
    ax.spines[['right', 'top']].set_visible(False)
    ax.xaxis.set_visible(False)
    

    ax=pl.subplot(gs[3,2])
    mean=N.mean(res['for decoding']['y_pred across binned right'],axis=0)
    sem=st.sem(res['for decoding']['y_pred across binned right'],axis=0)
    
    y1=mean-sem
    y2=mean+sem
    x=range(len(y1))
    ax.fill_between(x,y1,y2,alpha=0.2,color="green")
    ax.plot(mean,linewidth=1,color="k")
    ax.set_xticks([0,10,20])
    ax.set_xticklabels([0,0.5,1.0])
    ax.set_yticks([0,10,20])
    ax.set_yticklabels([0,0.5,1])
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel("True phase")
    ax.set_ylabel("Predicted phase")
    
    # Summary of decoding - across.
    
    ax=pl.subplot(gs[3,3])
    data=N.empty((2,9))
    data[0]=res['for decoding']['score across']
    data[1]=res['for decoding']['score across rand']
    boxplot_with_lines(ax,data,ylabel="Mean absolute error",ymin=0,ymax=1,use_legend=False)
    #ax.set_yticks([0,0.1,0.2,0.3,0.4])
    
    stat_res=stats.paired_comparison_two_groups(data[0],data[1])
    plot_siginficance_symbol_paired(ax,data,stat_res,height=0.3)
    
    
    
    
    #### Panel trial outcome decoding with temporal information.
    
    res=N.load("/Users/jonas/Documents/projects/rest_move_sequences/results_files/decoding_trial_outcome_samp_start.npy",allow_pickle=True).item()
    ax=pl.subplot(gs[2:4,5])
    
    score=res['score']
    score_shuff=res['score shuff']
    
    x=[1,2,4]
    score=res['score']
    y1=N.mean(score,axis=0)+st.sem(score,axis=0)
    y2=N.mean(score,axis=0)-st.sem(score,axis=0)
    ax.fill_between(x,y1,y2,alpha=0.2,color="g")
    ax.plot(x,N.mean(score,axis=0),"g",linewidth=2,label="data")
    for n in range(len(score)):
        ax.plot(x,score[n],c="grey",linewidth=0.5)
    
    score=res['score shuff']
    y1=N.mean(score,axis=0)+st.sem(score,axis=0)
    y2=N.mean(score,axis=0)-st.sem(score,axis=0)
    ax.fill_between(x,y1,y2,alpha=0.2,color="grey")
    ax.plot(x,N.mean(score,axis=0),"k",linewidth=2,label="shuffled")
    
       
    ax.spines[['right', 'top']].set_visible(False)
    #ax.xaxis.set_visible(False)
    ax.set_xlabel("Time from sampling onset (s)")
    ax.set_ylabel("Accuracy")
    ax.set_xticks([1,2,4])
    ax.set_xlim(0.7,4)
    ax.set_ylim(0.2,1.0)
    leg=ax.legend(loc="lower left",frameon=False,markerscale=1,alignment="left", labelspacing=0.1,handlelength=0.5)
    for line in leg.get_lines():    # iterate legend lines
        line.set_linewidth(4)
    
    score=res['score']
    score_shuff=res['score shuff']
    stat_res=stats.two_way_repeated_measures_anova_general(score,score_shuff)

    # Plot significance.
    offset=0.02
    if stat_res['ANOVA result']['p-GG-corr'][1]<0.05:
        if 0.01<stat_res['pairwise']['p corrected'][0]<0.05:
            ax.text(1,N.max(score[:,0])+offset,"*",ha="center", va="bottom")
        elif 0.001<stat_res['pairwise']['p corrected'][0]<0.01:
            ax.text(1,N.max(score[:,0])+offset,"**",ha="center", va="bottom")
        elif stat_res['pairwise']['p corrected'][0]<0.001:
            ax.text(1,N.max(score[:,0])+offset,"***",ha="center", va="bottom")
            
    if stat_res['ANOVA result']['p-GG-corr'][1]<0.05:
        if 0.01<stat_res['pairwise']['p corrected'][1]<0.05:
            ax.text(2,N.max(score[:,1])+offset,"*",ha="center", va="bottom")
        elif 0.001<stat_res['pairwise']['p corrected'][1]<0.01:
            ax.text(2,N.max(score[:,1])+offset,"**",ha="center", va="bottom")
        elif stat_res['pairwise']['p corrected'][1]<0.001:
            ax.text(2,N.max(score[:,1])+offset,"***",ha="center", va="bottom")
            
    if stat_res['ANOVA result']['p-GG-corr'][1]<0.05:
        if 0.01<stat_res['pairwise']['p corrected'][2]<0.05:
            ax.text(4,N.max(score[:,1])+offset,"*",ha="center", va="bottom")
        elif 0.001<stat_res['pairwise']['p corrected'][2]<0.01:
            ax.text(4,N.max(score[:,1])+offset,"**",ha="center", va="bottom")
        elif stat_res['pairwise']['p corrected'][2]<0.001:
            ax.text(4,N.max(score[:,1])+offset,"***",ha="center", va="bottom")
             


def plot_ed1():
    
    # Plot colorplot inset.
    start=2000
    end=5000
    animal=44
    res=N.load("/Users/jonas/Documents/projects/rest_move_sequences/results_files/trastermap_analysis_%s.npy" %animal,allow_pickle=True).item()

    fig=pl.figure(tight_layout=True)
    ax=fig.add_subplot()
    n_neurons=len(res['spks'])   
    p=ax.pcolormesh(res['spks'][res['isort']],cmap=cmap)
    p.set_clim(0,10)
    ax.set_xlim(start, end)
    ax.spines[['right', 'top','bottom','left']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    gs=gridspec.GridSpec(3,5)
    fig=pl.figure(tight_layout=True,figsize=[12,8])
    ### Panel A ###
    
    
        
    animal=44
    res=N.load("/Users/jonas/Documents/projects/rest_move_sequences/results_files/trastermap_analysis_%s.npy" %animal,allow_pickle=True).item()

    n_neurons=len(res['spks'])    
    ax=pl.subplot(gs[0,0:2])
    #p=ax.pcolormesh(res['spks'][res['isort']],cmap=cmap)
    ax.vlines(res['trial starts left'],0,n_neurons,color="w",linestyles="dashed",linewidth=0.5)
    ax.vlines(res['trial ends left'],0,n_neurons,color="w",linewidth=0.5)
    ax.vlines(res['trial starts right'],0,n_neurons,color="yellow",linestyles="dashed",linewidth=0.5)
    ax.vlines(res['trial ends right'],0,n_neurons,color="yellow",linewidth=0.5)
    ax.set_ylabel("Neurons")
    ax.set_xlabel("Time (s)")
    p.set_clim(0,10)
    ax.set_xlim(start, end)
    ax.set_ylim(0,n_neurons)
    ax.set_xticklabels([0,25,50,75,100,125,150])
    ax.set_title("Rastermap",fontsize=10)
    ax.spines[['right', 'top']].set_visible(False)
    
    res=N.load("/Users/jonas/Documents/projects/rest_move_sequences/results_files/trial_duration_analysis.npy",allow_pickle=True).item()

    
    
    ### Panel B ###
    
    ax=pl.subplot(gs[0,2])
   
    dur_left=res['trial duration left total']
    dur_right=res['trial duration right total']
   
    bins=50
    ax.hist(dur_left,bins=bins,color="k",label="left")
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel("Time sampling to reward (s)")
    ax.set_ylabel("Trial number")
    ax.set_xticks([0,5,10,15,20])
    ax.set_yticks([0,5,10,15,20])
    ax.set_title("left-going",fontsize=10)
    

    
    ax.hist(dur_right,bins=bins,color="m",label="right")
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel("Time sampling to reward (s)")
    ax.set_ylabel("Trial number")
    ax.set_xticks([0,5,10,15,20])
    ax.set_yticks([0,5,10,15,20])
    ax.set_title("right-going",fontsize=10)
    ax.legend(loc=[0.5,0.5],frameon=False,markerscale=10,alignment="left", labelspacing=0.1,handlelength=0.1)
    

    ### Panel B ###
    

    x=[5,6,7,9,13,20]
    expr_strength=[]
    res=N.load("/Users/jonas/Documents/projects/rest_move_sequences/results_files/sequence_detection_results_100_100.npy",allow_pickle=True).item()
    expr_strength=N.empty((6,len(res['expression strength'])))
    expr_strength[0]=res['expression strength']

    
    res=N.load("/Users/jonas/Documents/projects/rest_move_sequences/results_files/sequence_detection_results_120_120.npy",allow_pickle=True).item()
    expr_strength[1]=res['expression strength']

    res=N.load("/Users/jonas/Documents/projects/rest_move_sequences/results_files/sequence_detection_results_140_140.npy",allow_pickle=True).item()
    expr_strength[2]=res['expression strength']

    res=N.load("/Users/jonas/Documents/projects/rest_move_sequences/results_files/sequence_detection_results_180_180.npy",allow_pickle=True).item()
    expr_strength[3]=res['expression strength']

    res=N.load("/Users/jonas/Documents/projects/rest_move_sequences/results_files/sequence_detection_results_260_260.npy",allow_pickle=True).item()
    expr_strength[4]=res['expression strength']

    res=N.load("/Users/jonas/Documents/projects/rest_move_sequences/results_files/sequence_detection_results_400_400.npy",allow_pickle=True).item()
    expr_strength[5]=res['expression strength']

    
    expr_strength=expr_strength.T

    ax=pl.subplot(gs[0,3])

    
    y1=N.mean(expr_strength,axis=0)+st.sem(expr_strength,axis=0)
    y2=N.mean(expr_strength,axis=0)-st.sem(expr_strength,axis=0)
    ax.fill_between(x,y1,y2,alpha=0.2,color="grey")
    for n in range(len(expr_strength)):
        ax.plot(x,expr_strength[n],c="grey",linewidth=0.5)
    ax.plot(x,N.mean(expr_strength,axis=0),c="k",linewidth=2)
       
        
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel("Sequence length (s)")
    ax.set_ylabel("Template-match similarity")
    ax.set_xlim(5,20)
    ax.set_ylim(0,0.5)

    stat_res=stats.one_way_repeated_measures_anova_general(expr_strength)
        
    ### Panel C
    res=N.load("/Users/jonas/Documents/projects/rest_move_sequences/results_files/sequence_detection_results_random_timepoints.npy",allow_pickle=True).item()
    
   
    ax=pl.subplot(gs[1,2])
    bins=100
    bins2=50
    x=N.linspace(0,1,bins)
    x2=N.linspace(0,1,bins2)
    
    data=res['expression strength by mouse']
    data_by_mouse=N.empty((len(data),bins2))
    for n in range(len(data)):
        binned_react,_,_=st.binned_statistic(res['start pos list by mouse'][n],data[n],bins=bins2)    
        data_by_mouse[n]=binned_react
        ax.plot(x2,binned_react,color="grey",linewidth=0.5)
        
    binned_react,_,_=st.binned_statistic(res['start pos list total'],res['expression strength total'],bins=bins)    
    ax.plot(x,binned_react,color="k",linewidth=2)
    
    
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel("Normalized trial time")
    ax.set_ylabel("Template-match similarity")

    ax.set_ylim(0,0.6)
    #ax.set_yticks([0.4,0.6,0.8,1.0])
    
    plot_data=N.empty((2,len(data_by_mouse)))
    plot_data[0]=data_by_mouse.T[0] # Take the first bin
    plot_data[1]=data_by_mouse.T[int(bins2/2)] # Take the bin at 0.5
    
    ax=pl.subplot(gs[1,3])
    boxplot_with_lines(ax,plot_data,xlabel=['first','middle'],ylabel="Template-match similarity",ymin=0,ymax=0.6) 
    
    expression_strength_data_by_mouse=plot_data
    local_stats=stats.paired_comparison_two_groups(plot_data[0],plot_data[1])
    
    plot_siginficance_symbol_paired(ax,plot_data,local_stats,height=0.1)
    
    
    ### Panel D
    
    
    
    ax=pl.subplot(gs[1,0])
        
    data=res['reactivations per trial by mouse']
    data_by_mouse=N.empty((len(data),bins2))
    for n in range(len(data)):
        binned_react,_,_=st.binned_statistic(res['start pos list by mouse'][n],data[n],bins=bins2)    
        data_by_mouse[n]=binned_react
        ax.plot(x2,binned_react,color="grey",linewidth=0.5)
        
    binned_react,_,_=st.binned_statistic(res['start pos list total'],res['reactivations per trial total'],bins=bins)    
    ax.plot(x,binned_react,color="k",linewidth=2)   
    
    
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel("Normalized trial time")
    ax.set_ylabel("Reactivation proportion")

    ax.set_ylim(0,1.6)
    ax.set_yticks([0,0.4,0.8,1.2,1.6])
    
    plot_data=N.empty((2,len(data_by_mouse)))
    plot_data[0]=data_by_mouse.T[0] # Take the first bin
    plot_data[1]=data_by_mouse.T[int(bins2/2)] # Take the bin at 0.5
    
    ax=pl.subplot(gs[1,1])
    boxplot_with_lines(ax,plot_data,xlabel=['first','middle'],ylabel="Reactivation proportion",ymin=0,ymax=1.6) 
    ax.set_yticks([0,0.4,0.8,1.2,1.6])
    
    reactivation_data_by_mouse=plot_data
    local_stats=stats.paired_comparison_two_groups(plot_data[0],plot_data[1])
    
    plot_siginficance_symbol_paired(ax,plot_data,local_stats,height=0.1)
    
    
def plot_ed2():

    res=N.load("/Users/jonas/Documents/projects/rest_move_sequences/results_files/sequence_detection_results_100_100.npy",allow_pickle=True).item()
    fig=pl.figure(tight_layout=True,figsize=[10,8])
    gs=gridspec.GridSpec(5,5)
    
    speed_r=res['speed neuron r']
    speed_r_shuff=res['speed neuron r shuff']
    speed_r_by_mouse=res['speed neuron r by mouse']
    current_speed=res['current speed total']
    max_points_temporal=res['max points temporal']

    random_idx=N.random.choice(range(len(speed_r)),size=3,replace=False)
    
    
    ax=pl.subplot(gs[0,0])
    ax.plot(max_points_temporal[random_idx[0]],c="m",alpha=0.7)
    ax1=ax.twinx()
    ax1.plot(current_speed[random_idx[0]],c="gray")
    ax.spines[['top']].set_visible(False)
    ax1.spines[['top']].set_visible(False)
    ax.set_ylim(0,15)
    ax1.set_ylim(0,4)
    ax.set_ylabel("# active neurons",c="m",alpha=0.7)
    ax1.set_ylabel("Speed (a.u.)",c="gray")
    ax.set_xlabel("Sequence time (s)")
    ax.set_xticks([0,20,40,60,80,100])
    ax.set_xticklabels([0,1,2,3,4,5])
   
    
    ax=pl.subplot(gs[1,0])
    ax.plot(max_points_temporal[random_idx[1]],c="m",alpha=0.7)
    ax1=ax.twinx()
    ax1.plot(current_speed[random_idx[1]],c="gray")
    ax.spines[['top']].set_visible(False)
    ax1.spines[['top']].set_visible(False)
    ax.set_ylim(0,15)
    ax1.set_ylim(0,4)
    ax.set_ylabel("# active neurons",c="m",alpha=0.7)
    ax1.set_ylabel("Speed (a.u.)",c="gray")
    ax.set_xlabel("Sequence time (s)")
    ax.set_xticks([0,20,40,60,80,100])
    ax.set_xticklabels([0,1,2,3,4,5])
    
    
    ### Panel B ###
    ax=pl.subplot(gs[0:2,1])
    
    mean=N.mean(max_points_temporal,axis=0)
    sd=N.std(max_points_temporal,axis=0)
    
   
    y1=mean-sd
    y2=mean+sd
    x=range(len(y1))
    ax.fill_between(x,y1,y2,alpha=0.2,color="green")
    ax.plot(mean,linewidth=1,color="k")
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylim(0,15)
    ax.set_yticks([0,5,10,15])
    ax.set_xlim(-5,105)
    ax.set_xticks([0,20,40,60,80,100])
    ax.set_xticklabels([0,1,2,3,4,5])
    ax.set_ylabel("# active neurons")
    ax.set_xlabel("Sequence time (s)")
    
    
    # Asses significance by permutation.
    null_distribution=N.empty((permutation_iterations,len(max_points_temporal[0])))
    for n in range(permutation_iterations):
        shuffled_local=N.array([N.random.permutation(row) for row in max_points_temporal])
        shuffled_local_mean=N.mean(shuffled_local,axis=0)
        null_distribution[n]=shuffled_local_mean
    
    sig,perc=[],[]
    for n in range(len(mean)):
        perc.append(N.percentile(null_distribution.T[n],permutation_thres))
        if mean[n]>N.percentile(null_distribution.T[n],permutation_thres):
            sig.append(1)
        else:
            sig.append(0)
            
    ax.plot(perc,linewidth=1,color="gray",linestyle="dashed")
     
    print(N.ravel(N.argwhere(N.asarray(sig)>0.5)))
    
    ### Panel C ###

    ax=pl.subplot(gs[0:2,2])
    
    ax.plot(N.sort(speed_r),c="g",label="data")
    ax.plot(N.sort(speed_r_shuff),c="gray",label="shuffled")
    ax.legend(loc=[0,0],frameon=False,markerscale=1,alignment="left", labelspacing=0.1,handlelength=0.1)
    
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylim(-0.4,0.4)
    ax.set_xlim(0,300)
    ax.set_yticks([-0.4,-0.2,0,0.2,0.4])
    ax.set_xticks([0,100,200,300])
    ax.set_ylabel("Correlation # neurons x speed")
    ax.set_xlabel("Sorted trials")
    
    ax=pl.subplot(gs[0:2,3])
    data=N.empty((2,len(speed_r_by_mouse)))
    data[0]=speed_r_by_mouse
    data[1]=res['speed neuron r by mouse shuff']
    boxplot_with_lines(ax,data,xlabel=['data','shuffled'],colours=['green','gray'],ylabel="Correlation # neurons x speed",ymin=-0.4,ymax=0.2,use_legend=True)
    stat_res=stats.paired_comparison_two_groups(data[0],data[1])
    plot_siginficance_symbol_paired(ax,data,stat_res,height=0.08)  
    ax.set_ylim(-0.4,0.4)
    ax.set_yticks([-0.4,-0.2,0,0.2,0.4])
    

  
def plot_ed3():
    res=N.load("/Users/jonas/Documents/projects/rest_move_sequences/results_files/decoding_elapsed_time_samp_start.npy",allow_pickle=True).item()
    fig=pl.figure(tight_layout=True,figsize=[10,8])
    gs=gridspec.GridSpec(5,5)
    
    
    # Sampling
    ax=pl.subplot(gs[0,0])
    data=N.mean(res['cm'],axis=0)
    p=ax.pcolormesh(data,cmap=cmap)  
    p.set_clim(0,0.5)
    p.set_clim(0,0.5)
    ax.set_xticks([0,10,20])
    ax.set_xticklabels([0,0.5,1.0])
    ax.set_yticks([0,10,20])
    ax.set_yticklabels([0,0.5,1.0])
    ax.set_xlabel("True time")
    ax.set_ylabel("Pred. time")
    ax.set_title("data",fontsize=10)
    
    ax=pl.subplot(gs[1,0])
    data=N.mean(res['cm shuff'],axis=0)
    p=ax.pcolormesh(data,cmap=cmap)  
    p.set_clim(0,0.5)
    ax.set_xticks([0,10,20])
    ax.set_xticklabels([0,0.5,1.0])
    ax.set_yticks([0,10,20])
    ax.set_yticklabels([0,0.5,1.0])
    ax.set_xlabel("True time")
    ax.set_ylabel("Pred. time")
    ax.set_title("shuffled",fontsize=10)
    
    
    ax=pl.subplot(gs[0:2,1])  
    
    data=N.empty((2,len(res['score'])))
    data[0]=res['score']
    data[1]=res['score shuff']
    
    boxplot_with_lines(ax,data,xlabel=['data','shuffled'],colours=['green','gray'],ylabel="Mean absolute error (frames)",ymin=0,ymax=8,use_legend=True)
    stat_res=stats.paired_comparison_two_groups(data[0],data[1])
    plot_siginficance_symbol_paired(ax,data,stat_res,height=0.08)
    
    
    # Rew
    res=N.load("/Users/jonas/Documents/projects/rest_move_sequences/results_files/decoding_elapsed_time_rew_start.npy",allow_pickle=True).item()
    
    ax=pl.subplot(gs[0,2])
    data=N.mean(res['cm'],axis=0)
    p=ax.pcolormesh(data,cmap=cmap)  
    p.set_clim(0,0.5)
    p.set_clim(0,0.5)
    ax.set_xticks([0,10,20])
    ax.set_xticklabels([0,0.5,1.0])
    ax.set_yticks([0,10,20])
    ax.set_yticklabels([0,0.5,1.0])
    ax.set_xlabel("True time")
    ax.set_ylabel("Pred. time")
    ax.set_title("data",fontsize=10)
    
    ax=pl.subplot(gs[1,2])
    data=N.mean(res['cm shuff'],axis=0)
    p=ax.pcolormesh(data,cmap=cmap)  
    p.set_clim(0,0.5)
    ax.set_xticks([0,10,20])
    ax.set_xticklabels([0,0.5,1.0])
    ax.set_yticks([0,10,20])
    ax.set_yticklabels([0,0.5,1.0])
    ax.set_xlabel("True time")
    ax.set_ylabel("Pred. time")
    ax.set_title("shuffled",fontsize=10)
    
    
    ax=pl.subplot(gs[0:2,3])  
    
    data=N.empty((2,len(res['score'])))
    data[0]=res['score']
    data[1]=res['score shuff']
    
    boxplot_with_lines(ax,data,xlabel=['data','shuffled'],colours=['green','gray'],ylabel="Mean absolute error (frames)",ymin=0,ymax=8,use_legend=True)
    stat_res=stats.paired_comparison_two_groups(data[0],data[1])
    plot_siginficance_symbol_paired(ax,data,stat_res,height=0.08)
   


#plot_figure1()  
#plot_figure2() 

#plot_ed1()
#plot_ed2()  
plot_ed3()  

        
        