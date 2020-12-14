#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import sys
import rawpy
import png
import glob2
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display


# In[2]:


def readImagesAndTimes(filenames):
    images = []
    for filename in filenames:
        raw = rawpy.imread(filename)
        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        images.append(im)
    return images


# In[4]:


def ExposureFusion(filenames):
    
    # Read example images
    images = readImagesAndTimes(filenames)
    
    mergeMertens = cv2.createMergeMertens()
    exposureFusion = mergeMertens.process(images)
    

    # Convert gt_full to 16 bit unsigned integers.
    z = (65535*((exposureFusion - exposureFusion.min())/exposureFusion.ptp())).astype(np.uint16)
    
    
    with open('result.png', 'wb') as f:
        writer = png.Writer(width=z.shape[1], height=z.shape[0], bitdepth=16, greyscale=False)
        
        # Convert z to the Python list of lists expected by
        # the png writer.
        z2list = z.reshape(-1, z.shape[1]*z.shape[2]).tolist()
        writer.write(f, z2list)


# In[8]:


raw_dir = 'Example_images/RAW/'

raw_fns = glob2.glob(raw_dir + '*.ARW')

filenames = []
filenames.append(raw_fns.pop())

ExposureFusion(filenames)

image = Image.open('result.png')
image_array = np.asarray(image)
plt.imshow(image_array)
plt.show()


# In[ ]:




