#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:24:22 2026

@author: jonas
"""

'''
Helper functions for task-progress sequences analysis.

'''

import numpy as N

def rescale_pos(values, new_min = 0, new_max= 1):
    '''

    Parameters
    ----------
    values : 1d array
    new_min : int, optional
        New min value. The default is 0.
    new_max : int, optional
        New max value. The default is 1.

    Returns
    -------
    output : 1d array
        Values rescaled.
    '''

    output = []
    old_min, old_max = min(values), max(values)

    for v in values:
        new_v = (new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min
        output.append(new_v)

    return N.asarray(output)

def rescale_pos_2d(values, new_min = 0, new_max= 1):
    '''

    Parameters
    ----------
    values : 2d array
    new_min : int, optional
        New min value. The default is 0.
    new_max : int, optional
        New max value. The default is 1.

    Returns
    -------
    output : 2d array
        Values rescaled.

    '''
    output = N.empty_like(values)
    
    for n in range(len(values)):
        
        new_v = rescale_pos(values[n],new_min=new_min,new_max=new_max)
        output[n]=new_v

    return output

def get_spatial_map_sorting_indices(data):
    '''

    Parameters
    ----------
    data : 2d array
        neurons x time points.

    Returns
    -------
    Sorted indices according to time of peak.

    '''
    max_points=N.argmax(data,axis=1)
    idx=N.argsort(max_points)
    return idx

def get_spatial_map_peak_locations(data):
    '''

    Parameters
    ----------
    data : 2d array
        neurons x time points.

    Returns
    -------
    Array with time of peak. for each neuron

    '''
    max_points=N.argmax(data,axis=1)
    return max_points

def separate_in(indices,is_in):
    ''' Separate 2 periods of spike-like (point) data.
    '''    
   
    is_in_range=[]
    for n in range(len(is_in[0::2])):
        is_in_range.extend(range(is_in[0::2][n],is_in[1::2][n],1))
    is_in_range=set(is_in_range)
  
    ind=[]
    for n in range(len(indices)):
        if indices[n] in is_in_range:
            ind.append(indices[n])
   
    return N.asarray(ind,dtype="int")


        