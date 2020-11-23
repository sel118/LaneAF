#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import paf_test as paf
import cv2
import numpy as np
import glob

def generate_pafs():
    glob_pattern = os.path.join('.', 'data', 'seg_label', '*', '*', '*.png')
    image_filenames = sorted(glob.glob(glob_pattern))
    for i in image_filenames:
        label = cv2.imread(i)
        split_name = i.split('/')
        generatedPAFs = paf.generatePAFs(label[:, :, 0], viz=False)
        np.save(os.path.join(split_name[0], split_name[1], split_name[2], split_name[3], split_name[4], '20.npy'), generatedPAFs)