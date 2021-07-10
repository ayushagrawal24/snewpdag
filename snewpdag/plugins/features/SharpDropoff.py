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
    [t_check_width]: (int/float, optional) time difference in ms over which drop in luminosity is checked. Assumed to be evenly spaced
    [threshold_drop]: (float, optional) 1 - minimum percentage drop in luminosity over dropWidth to flag as a BH candidate. Must be specified with dropWidth. (eg drop threshold from 100 to 3 over t_check_width would be 0.03)
    
Output:
    has_sharp_drop: (bool) whether or not a sharp drop was detected
    drop_times: (1d numpy array of ints) Time bin at which start of a sharp drop is seen. Ideally expect only a single drop
    drop_diffs: (1d numpy array of floats) difference in neutrino count at drop

"""

import logging
import numpy as np
import healpy as hp
from snewpdag.dag import Node

class SharpDropoff(Node):
    
  #Constructor
  def __init__(self, in_field, **kwargs):
    self.tfield = in_xfield
    self.yfield = in_yfield
    self.t_check_width = kwargs.pop('t_check_width', 10)
    self.threshold_drop = kwargs.pop('threshold_drop', 0.1)
    self.out_field = kwargs.pop('out_field', None)
    super().__init__(**kwargs)

  #
  def alert(self, data):
    times = np.array(data[self.tfield])
    vals = np.array(data[self.yfield])

    #Finding indices of t+t_check_width for each t
    #This is needed in case the width is not an integer multiple of the time bin spacing
    #Can lead to an error of 1 in indices due to floating point error
    ind = np.searchsorted(times, times + t_check_width, side='left')
    last_ind = np.argmax(ind)
    condition = vals[ind[:last_ind]] - vals[:last_ind] > self.threshold_drop

    #If width guaranteed to be an integer multiple of bin spacing
    ind_width = ceil((times[1]-times[0])/self.t_check_width)
    condition = np.diff(vals, ind_width) > self.threshold_drop

    #Report only the starting point of any detected drops
    drop_ind = np.nonzero(condition)
    drop_ind = drop_ind[np.insert(np.diff(condition) - 1 == 0, 0, True] 
                                    
    self.drop_times = times[drop_ind]
    self.found_dropoff = bool(self.drop_times.size)
    
    return False

  def reset(self, data):
    return False

  def revoke(self, data):
    return False

  def report(self, data):
    d = {
          "has_sharp_drop": self.found_dropoff,
          "drop_times": self.drop_times,
        }
    if self.out_field == None:
      data.update(d)
    else:
      data[self.out_field] = d
    return True
