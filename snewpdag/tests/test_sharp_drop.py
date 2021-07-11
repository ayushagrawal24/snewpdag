# -*- coding: utf-8 -*-
import unittest
import numpy as np
import json
from snewpdag.plugins.Features import SharpDropoff as sd

class TestSharpDropoff(unittest.TestCase):
  def test_alert(self):
    with open('snewpdag/data/test-flux-input.json', 'r') as f:
      #Only testing with KM3Net data for now
      self.data = json.load(f)[1]
      
    sd_plugin = sd.SharpDropoff('t_low', 't_bins', name=self.data["name"])
    sd_plugin.alert(self.data)
    
    self.assertFalse(self.data['has_drop'])