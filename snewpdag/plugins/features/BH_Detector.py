#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 00:15:00 2021

@author: ayushagrawal
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 5 2021

@author: ayushagrawal

SharpDropoff : Looks for a sharp drop in the signal, indicating a potential black hole. Currently assumes binned input data from a single experiment

Input Arguments:
    in_xfield: (string) name of field to extract from alert data containing time bins
    in_yfield: (string) name of field to extract from alert data containing neutrino detection counts
    
    [char_time]: (float, optional) characteristic time scale expected for the drop (in s)
    [penalty]: (float, optional) Penalty (beta) parameter to use in PELT model
    [thresh_drop]: (float, optional) Minimum required factor by which signal is required to drop 
    
Output:
    has_drop: (bool) whether or not a sharp drop was detected
    drop_time: (float) Time bin at which the start of the sharpest drop is seen.
    max_drop: (float) Sharpest drop value

"""

import logging
import numpy as np
import scipy.ndimage
from snewpdag.dag import Node

class BH_Detector(Node):
    
  #Constructor
  def __init__(self, in_xfield, in_yfield, **kwargs):
    self.tfield = in_xfield
    self.yfield = in_yfield
    
    self.epsilon = kwargs.pop('epsilon',60)      
    self.thresh_slope = kwargs.pop('threshold_slope', -400)
    self.out_field = kwargs.pop('out_field', None)
    super().__init__(**kwargs)
    
    self.drop_time = None
    self.found_drop = False
    logging.basicConfig(level=logging.INFO)
    
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
    
    
  def canny_edge_detector(times, vals, strong_threshold = 95, weak_threshold = 80):
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
    drop_ind = self.d2_edge_finder(times, vals)
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