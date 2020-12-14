import cv2
import numpy as np
import sys
import rawpy
import png
import glob

def readImagesAndTimes(filenames):
  images = []
  for filename in filenames:
    raw = rawpy.imread(filename)
    im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    #im = im/1.0               #Change the factor to adjust brightness
    #print(filename)
    images.append(im)
  return images


def ExposureFusion(filenames,i):
  # Read images
  print('Group number %d'%i)
  print("    Reading images ... ")
  # Read example images
  images = readImagesAndTimes(filenames)
  # Can't Align input images, so skip that step
  
  # Merge using Exposure Fusion
  print("    Merging using Exposure Fusion ... ")
  mergeMertens = cv2.createMergeMertens()
  exposureFusion = mergeMertens.process(images)

  # Convert gt_full to 16 bit unsigned integers.
  z = (65535*((exposureFusion - exposureFusion.min())/exposureFusion.ptp())).astype(np.uint16)
  # Save output image
  print("    Saving output...")
  with open('result/00%d_00_16s.png'%(i+219), 'wb') as f:
    writer = png.Writer(width=z.shape[1], height=z.shape[0], bitdepth=16)
    # Convert z to the Python list of lists expected by
    # the png writer.
    z2list = z.reshape(-1, z.shape[1]*z.shape[2]).tolist()
    writer.write(f, z2list)
  


if __name__ == '__main__':
  
  raw_dir = './OurDataset/raw/part6/'
  raw_fns = glob.glob(raw_dir + '*.ARW')
  
  i = 0
  for i in range(999):
    if not raw_fns:
      break
    j = 0
    filenames = []
    for j in range(9):
      filenames.append(raw_fns.pop())
      j += 1
    ExposureFusion(filenames,i)
    i += 1
    
