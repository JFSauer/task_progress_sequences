#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:47:17 2025

@author: jonas

Collection of statistical functions for task_progress_sequences


"""
import numpy as N
import pandas as pd
import pingouin as pg
import scipy.stats as st
from itertools import permutations

def two_way_repeated_measures_anova_general(data1,data2,use_ttest=False):

    #Data arrays given as data (mice, cells) x time points
    data1=N.asarray(data1)
    data2=N.asarray(data2)
    data=data1
    data=N.append(data1,data2,axis=0)
    data=N.ravel(data)
    condition=N.zeros((len(N.ravel(data1))))      
    condition=N.append(condition,N.ones((len(N.ravel(data1))))) 
    
    mouse=N.repeat(N.linspace(1,len(data1),len(data1)),len(data1[0]))
    mouse=N.append(mouse,mouse,axis=0)
    
    time=N.tile(N.linspace(1,len(data1[0]),len(data1[0])),len(data1)*2)
    
    df=pd.DataFrame({'mouse':mouse,
                    'time':time,
                    'condition':condition,
                    'data':data})    
    
    res = pg.rm_anova(dv='data', within=['time','condition'], subject='mouse', 
                  data=df, detailed=True)
    
    # Post hoc comparisons with paired t-tests and Sidak correction.
    data1=data1.T
    data2=data2.T
    t,p=[],[]
    sig=[]
    test_used=[]
    pcrit=1-(1-0.05)**(1/len(data1))
    p_corr=[]
    for n in range(len(data1)):
        temp=paired_comparison_two_groups(data1[n],data2[n],use_ttest=use_ttest)
        #tt,pp=st.ttest_rel(data1[n],data2[n])
        t.append(temp['statistic'])
        test_used.append(temp['test'])
        pp=temp['p']
        p.append(pp)
        p_corr.append(pp*0.05/pcrit)
        if pp<pcrit:
            sig.append(1)
        else:
            sig.append(0)
    
    #t=df.pairwise_tukey(dv='data',between='condition').round(3)
    pairwise={'p':p,'t':t,'sig':sig,'Pcrit':pcrit,'p corrected':p_corr,'test used':test_used}

    res_total={'ANOVA result':res.to_dict(),'pairwise':pairwise}
   
    

    return res_total

def one_way_repeated_measures_anova_general(data1):

    #Data arrays given as data (mice, cells) x time points

   
    mouse=N.repeat(N.linspace(1,len(data1),len(data1)),len(data1[0]))
    
    time=N.tile(N.linspace(1,len(data1[0]),len(data1[0])),len(data1))
    data=N.ravel(data1)

    
    df=pd.DataFrame({'mouse':mouse,
                    'time':time,
                    'data':data})    
    
    res = pg.rm_anova(dv='data', within='time', subject='mouse', 
                  data=df, detailed=True)
      
    # Post hoc comparisons with paired tests and Sidak correction.
    data=data1.T
    t,p,test=[],[],[]
    sig=[]
    comp=[]
    
    p_corr=[]
    counter=0
    for n in range(len(data)-1):
        for i in range(n+1,len(data),1):
            counter+=1
            local_comp=[n, i]
            comp.append(local_comp)
            local=paired_comparison_two_groups(data[n],data[i])
            tt,pp=st.ttest_rel(data[n],data[i])
            t.append(local['statistic'])
            p.append(local['p'])
            test.append(local['test'])
    
    pcrit=1-(1-0.05)**(1/counter)       
    for n in range(len(data)-1):
        for i in range(n+1,len(data),1): 
            local=paired_comparison_two_groups(data[n],data[i])
            tt,pp=st.ttest_rel(data[n],data[i])
            
            p_corr.append(pp*0.05/pcrit)
            if local['p']<pcrit:
                sig.append(1)
            else:
                sig.append(0)
    

    res_total={'p':p,'t':t,'sig':sig,'Pcrit':pcrit,'Pcorr':p_corr,'comparisons':comp,'ANOVA':res.to_dict(),'post hoc test':test}
    return res_total

def one_way_repeated_measures_anova_general_three_groups(array1,array2,array3):

    #Data arrays given as data (mice, cells) x time points

    temp=[]
    temp.extend(array1)
    temp.extend(array2)
    temp.extend(array3)
    data1=N.reshape(temp,[3,-1]).T
   
    
    mouse=N.repeat(N.linspace(1,len(data1),len(data1)),len(data1[0]))
    
    time=N.tile(N.linspace(1,len(data1[0]),len(data1[0])),len(data1))
    data=N.ravel(data1)

    
    df=pd.DataFrame({'mouse':mouse,
                    'time':time,
                    'data':data})    
    
    res = pg.rm_anova(dv='data', within='time', subject='mouse', 
                  data=df, detailed=True)
    
    # Post hoc comparisons with paired t-tests and Sidak correction.
    data=data1.T
    t,p,test=[],[],[]
    sig=[]
    comp=[]
    pcrit=1-(1-0.05)**(1/len(data))
    p_corr=[]
    for n in range(len(data)-1):
        for i in range(n+1,len(data),1):
            local_comp=[n, i]
            comp.append(local_comp)
            local=paired_comparison_two_groups(data[n],data[i])
            tt,pp=st.ttest_rel(data[n],data[i])
            t.append(local['statistic'])
            p.append(local['p'])
            test.append(local['test'])
            p_corr.append(pp*0.05/pcrit)
            if local['p']<pcrit:
                sig.append(1)
            else:
                sig.append(0)
    

    res_total={'p':p,'t':t,'sig':sig,'Pcrit':pcrit,'Pcorr':p_corr,'comparisons':comp,'ANOVA':res.to_dict(),'post hoc test':test}
    

    return res_total

def one_way_repeated_measures_anova_general_four_groups(array1,array2,array3,array4):

    #Data arrays given as data (mice, cells) x time points

    temp=[]
    temp.extend(array1)
    temp.extend(array2)
    temp.extend(array3)
    temp.extend(array4)
    data1=N.reshape(temp,[4,-1]).T
   
    
    mouse=N.repeat(N.linspace(1,len(data1),len(data1)),len(data1[0]))
    
    time=N.tile(N.linspace(1,len(data1[0]),len(data1[0])),len(data1))
    data=N.ravel(data1)

    
    df=pd.DataFrame({'mouse':mouse,
                    'time':time,
                    'data':data})    
    
    res = pg.rm_anova(dv='data', within='time', subject='mouse', 
                  data=df, detailed=True)
    
    # Post hoc comparisons with paired t-tests and Sidak correction.
    data=data1.T
    t,p,test=[],[],[]
    sig=[]
    comp=[]
    pcrit=1-(1-0.05)**(1/6)
    p_corr=[]
    for n in range(len(data)-1):
        for i in range(n+1,len(data),1):
            local_comp=[n, i]
            comp.append(local_comp)
            local=paired_comparison_two_groups(data[n],data[i])
            tt,pp=st.ttest_rel(data[n],data[i])
            t.append(local['statistic'])
            p.append(local['p'])
            test.append(local['test'])
            p_corr.append(pp*0.05/pcrit)
            if local['p']<pcrit:
                sig.append(1)
            else:
                sig.append(0)
    

    res_total={'p':p,'t':t,'sig':sig,'Pcrit':pcrit,'Pcorr':p_corr,'comparisons':comp,'ANOVA':res.to_dict(),'post hoc test':test}
    

    return res_total

def one_way_repeated_measures_anova_general_five_groups(array1,array2,array3,array4,array5):

    #Data arrays given as data (mice, cells) x time points

    temp=[]
    temp.extend(array1)
    temp.extend(array2)
    temp.extend(array3)
    temp.extend(array4)
    temp.extend(array5)
    data1=N.reshape(temp,[5,-1]).T
   
    
    mouse=N.repeat(N.linspace(1,len(data1),len(data1)),len(data1[0]))
    
    time=N.tile(N.linspace(1,len(data1[0]),len(data1[0])),len(data1))
    data=N.ravel(data1)

    
    df=pd.DataFrame({'mouse':mouse,
                    'time':time,
                    'data':data})    
    
    res = pg.rm_anova(dv='data', within='time', subject='mouse', 
                  data=df, detailed=True)
    
    # Post hoc comparisons with paired t-tests and Sidak correction.
    data=data1.T
    t,p,test=[],[],[]
    sig=[]
    comp=[]
    pcrit=1-(1-0.05)**(1/10)
    p_corr=[]
    for n in range(len(data)-1):
        for i in range(n+1,len(data),1):
            local_comp=[n, i]
            comp.append(local_comp)
            local=paired_comparison_two_groups(data[n],data[i])
            t.append(local['statistic'])
            p.append(local['p'])
            test.append(local['test'])
            p_corr.append(local['p']*0.05/pcrit)
            if local['p']<pcrit:
                sig.append(1)
            else:
                sig.append(0)
    

    res_total={'p':p,'t':t,'sig':sig,'Pcrit':pcrit,'Pcorr':p_corr,'comparisons':comp,'ANOVA':res.to_dict(),'post hoc test':test}
    

    return res_total

def paired_comparison_two_groups(data1,data2,use_ttest=False):
    
    data1=N.asarray(data1)
    data2=N.asarray(data2)
    tt,norm_check1=st.shapiro(data1)
    tt,norm_check2=st.shapiro(data2)
    if use_ttest==True:
        t,p=st.ttest_rel(data1,data2)
        used_test="t test"
    else:
        if norm_check1>0.05 and norm_check2>0.05:
            t,p=st.ttest_rel(data1,data2)
            used_test="t test"
        else:
            t,p=st.wilcoxon(data1,data2)
            used_test="Wilcoxon"
        
    res={'p':p,'statistic':t,'test':used_test}
    return res


def paired_permutation_test(data1,data2,statistic='mean'):
    
    diff=data1-data2
    n=len(data1)
    
    # Combined list.
    data=N.append(data1,data2)
    
    iterations=[]
    for n in permutations(data):
        iterations.append(n)
    return iterations
    
    
def unpaired_comparison_two_groups(data1,data2):
    
    data1=N.asarray(data1)
    data2=N.asarray(data2)
    tt,norm_check1=st.shapiro(data1)
    tt,norm_check2=st.shapiro(data2)
    if norm_check1>0.05 and norm_check2>0.05:
        t,p=st.ttest_ind(data1,data2)
        used_test="t test"
    else:
        t,p=st.mannwhitneyu(data1,data2)
        used_test="MWU"
        
    res={'p':p,'statistic':t,'test':used_test}
    return res
        
