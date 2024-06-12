# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:33:19 2024

@author: mark.patmore
"""

import numpy as np


inputs = [1, 2, 3, 2.5]
weights = [[.2, .8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

layer_outputs = np.dot(weights, inputs) + biases

print(layer_outputs)