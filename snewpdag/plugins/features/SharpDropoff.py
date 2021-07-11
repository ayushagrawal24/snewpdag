#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 5 2021

@author: ayushagrawal

SharpDropoff : Looks for a sharp drop in the signal, indicating a potential black hole. Currently assumes binned input data from a single experiment

Input Arguments:
    in_xfield: (string) name of field to extract from alert data containing time bins
    in_yfield: (string) name of field to extract from alert data containing neutrino detection counts
    
    [out_field]: (string, optional) name of field for dictionary output.
    [char_time]: (float, optional) characteristic time scale expected for the drop (in s)
    [penalty]: (float, optional) Penalty (beta) parameter to use in PELT model
    [thresh_drop]: (float, optional) Minimum required factor by which signal is required to drop 
    
Output:
    has_sharp_drop: (bool) whether or not a sharp drop was detected
    drop_time: (float) Time bin at which the start of the sharpest drop is seen.
    max_drop: (float) Sharpest drop value

"""

import logging
import math
import numpy as np
import ruptures as rpt
from snewpdag.dag import Node

class SharpDropoff(Node):
    
  #Constructor
  def __init__(self, in_xfield, in_yfield, **kwargs):
    self.found_dropoff = False
    self.tfield = in_xfield
    self.yfield = in_yfield
    self.cpd = rpt.KernelCPD(kernel='linear')
    self.char_time = kwargs.pop('char_time',0.0002)
    self.cpd_penalty = kwargs.pop('penalty',100)      
    self.thresh_drop = kwargs.pop('threshold_drop', 3) 
    self.out_field = kwargs.pop('out_field', None)
    super().__init__(**kwargs)

  def alert(self, data):
    times = data[self.tfield]
    dt = times[1]-times[0]
    vals = data[self.yfield]
    log_vals = np.log2(vals)

    #Minimum segment size for change point detection
    self.cpd.min_size = min(2, math.ceil(self.char_time/dt))

    #find changepoint indices and prepend start point 0
    bkps = self.cpd.fit_predict(signal=log_vals, pen=self.cpd_penalty)
    bkps = np.concatenate(([0],bkps))
    
    log_means = np.zeros(bkps.size-1)
    for i in range(bkps.size-1):
      log_means[i] = log_vals[bkps[i]:bkps[i+1]].mean()
      
    self.drop_max = np.diff(log_means).max()
    logging.info('Sharpest drop: {}'.format(self.drop_max))
    
    if self.drop_max <= -1*math.log2(self.thresh_drop):
      self.drop_time = times[np.diff(log_means).argmax()]
      self.found_dropoff = True
      logging.info('Potential BH formation at time {}'.format(self.drop_time))
    else:
      logging.info('No BH candidate detected')
    
    logging.info('BH drop detected: {}'.format(self.found_dropoff))
      
    d = {
          "has_sharp_drop": self.found_dropoff,
          "drop_time": self.drop_time,
          "max_drop": 2**self.drop_max
        }
    
    if self.out_field == None:
      data.update(d)
    else:
      data[self.out_field] = d
    
    return True