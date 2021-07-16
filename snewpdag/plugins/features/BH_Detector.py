#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 2021

@author: ayushagrawal24

BH_Detector : Looks for a sharp drop in the signal, indicating a potential black hole. Currently assumes evenly binned input data from a single experiment

Input Arguments:
    in_xfield: (string) name of field to extract from alert data containing time bins
    in_yfield: (string) name of field to extract from alert data containing neutrino detection counts
    
    
        
Output:
    has_drop: (bool) whether or not a sharp drop was detected
    drop_time: (float) Time bin at which the start of the sharpest drop is seen.

"""

import logging
import numpy as np
import scipy.ndimage
from snewpdag.dag import Node

import ruptures as rpt
import math

class BH_Detector(Node):
    
  #Constructor
  def __init__(self, in_xfield, in_yfield, **kwargs):
    self.tfield = in_xfield
    self.yfield = in_yfield
    
    #Canny params
    self.epsilon = kwargs.pop('epsilon',60)      
    self.thresh_slope = kwargs.pop('threshold_slope', -400)
    #d2_edge_finder params
    self.char_time = kwargs.pop('char_time',0.0002)
    self.cpd_penalty = kwargs.pop('penalty',0.6)      
    self.thresh_drop = kwargs.pop('threshold_drop', 1.7) 
    self.cpd = rpt.KernelCPD(kernel='linear')
        
    self.drop_time = None
    self.found_drop = False
    logging.basicConfig(level=logging.INFO)   
    
    super().__init__(**kwargs)
    
  def cpd_edge_finder(self, times, vals):
    times = np.asarray(times)
    dt = times[1]-times[0]
    log_vals = np.log2(vals)

    #Minimum segment size for change point detection
    self.cpd.min_size = max(2, math.ceil(self.char_time/dt))
    
    #find changepoint indices and prepend start point 0
    bkps = self.cpd.fit_predict(signal=log_vals, pen=self.cpd_penalty)
    bkps = np.concatenate(([0],bkps))
    bkp_times = times[bkps[:-1]]
    logging.info('bkp_times: {}'.format(bkp_times))
    
    log_means = np.zeros(bkps.size-1)
    std_dev = np.zeros(bkps.size-1)
    for i in range(bkps.size-1):
      log_means[i] = log_vals[bkps[i]:bkps[i+1]].mean()
      std_dev[i] = np.exp2(log_vals[bkps[i]:bkps[i+1]]).std()
      
    logging.info('means: {}'.format(np.exp2(log_means)))
    logging.info('std_dev: {}'.format(std_dev))
      
    self.drop_max = -1*np.diff(log_means).min()
    logging.info('Sharpest_drop: {}'.format(self.drop_max))
    
    if self.drop_max >= math.log2(self.thresh_drop):
      self.drop_time = bkp_times[np.diff(log_means).argmin()]
      self.found_dropoff = True
      logging.info('BH_time: {}'.format(self.drop_time))
    
    
  def d2_edge_finder(self, times, vals, edge_type='falling'):
    #Smoothen curve with median filter
    smooth = scipy.ndimage.median_filter(vals, size=5)

    #find first and second derivatives
    dt = (times[-1]-times[0])*1000/len(times)
    diff1 = np.convolve(smooth, [1, 0, -1], mode='same')/dt
    diff2 = np.convolve(diff1, [1, -1],mode='same')/dt
    
    #second derivative 0, changes sign from left to right
    diff_extrema = np.logical_and(np.abs(diff2)[1:-1] < self.epsilon, np.sign(diff2[:-2])*np.sign(diff2[2:]) < 0)
        
    if edge_type == 'falling':
      edge_ind = np.where( np.logical_and.reduce((diff_extrema, diff1[1:-1] < self.thresh_slope, diff2[:-2] < 0)) )[0]
    elif edge_type == 'rising':
      edge_ind = np.where( np.logical_and.reduce((diff_extrema, diff1[1:-1] > self.thresh_slope, diff2[:-2] > 0)) )[0]
    elif edge_type == 'both':
      edge_ind = np.where( np.logical_and(diff_extrema, np.abs(diff1[1:-1]) > self.thresh_slope) )[0]
    else:
      edge_ind = np.array([])
      
    logging.info(edge_ind)
      
    return edge_ind + 1
  
  
  def canny_edge_detector(times, vals, strong_threshold = 95, weak_threshold = 90):
    #Smoothen curve with median filter
    smooth = scipy.ndimage.median_filter(vals, size=7)

    #find first derivative, scaled to 100
    diff = np.abs(np.convolve(smooth, [1, 0, -1], mode='same'))
    diff[0] = 0
    diff[-1] = 0
    diff = diff/diff.max()*100
    
    #Non-max suppression (edge thinning)
    edges = np.zeros(diff.shape)
    edges[1:-1] = np.where( np.logical_and(diff[1:-1] >= diff[:-2], diff[1:-1] >= diff[2:]), diff[1:-1], 0)
    
    #double threshold
    weak_i = np.where( np.logical_and(edges >= weak_threshold, edges < strong_threshold) )[0]
    strong_i = np.where( edges >= strong_threshold )[0]
    
    #Hysteresis edge tracking
    weak_i = weak_i[np.logical_or( np.in1d(weak_i+1,strong_i), np.in1d(weak_i-1, strong_i) )]
    strong_i = np.insert(strong_i, np.searchsorted(strong_i, weak_i), weak_i)
       
    return strong_i

  def alert(self, data):
    times = data[self.tfield]
    vals = data[self.yfield]
    
    #Find index of last falling edge
    drop_ind = self.d2_canny_edge_detector(times, vals)
    logging.info(drop_ind)

    if drop_ind.size == 0:
      self.found_drop = False
      self.drop_time = None
    else:
      self.drop_time = times[drop_ind[-1]]
      self.found_drop = True
      
    d = {
          "has_drop": self.found_drop,
          "drop_time": self.drop_time,
        }
    
    logging.info('{}'.format(d))
    
    data.update(d)
    
    return True