# -*- coding: utf-8 -*-
import unittest
import numpy as np
import json
import logging
from snewpdag.plugins.Features import SharpDropoff as sd

class TestSharpDropoff(unittest.TestCase):
  def setUp(self):
    with open('snewpdag/data/test-flux-input.json', 'r') as f:
      #Only testing with KM3Net data for now
      self.data = json.load(f)[1]
      
    self.sd_plugin = sd.SharpDropoff('t_low', 't_bins', name=self.data["name"])
    
  def test_alert_false(self):
    self.sd_plugin.alert(self.data)
    
    self.assertFalse(self.data['has_drop'])
    logging.info('test_alert_false success: {} \n\n'.format(self.data['has_drop'] == False))
  
    
  def test_alert_true(self):
    vals = self.data['t_bins']    
    vals[2260] = 1800
    vals[2261:6000] = np.random.normal(1555, 40, 6000 - 2261)
    
    
    self.data['t_bins'] = vals
    
    self.sd_plugin.alert(self.data)
    logging.info('test_alert_true success: {} \n\n'.format(self.data['has_drop'] == True))
    self.assertTrue(self.data['has_drop'])
    