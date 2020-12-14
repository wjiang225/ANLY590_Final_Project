import cv2
import numpy as np
import sys
import rawpy
import png

def readImagesAndTimes():
  
  filenames = [                  #Enter file names here
              "Example Images/RAW/DSC00353.ARW",
              "Example Images/RAW/DSC00354.ARW",
	      "Example Images/RAW/DSC00355.ARW",
	      "Example Images/RAW/DSC00356.ARW",
	      "Example Images/RAW/DSC00357.ARW",
              "Example Images/RAW/DSC00358.ARW",
              "Example Images/RAW/DSC00359.ARW",
              "Example Images/RAW/DSC00360.ARW",
              "Example Images/RAW/DSC00361.ARW"
               ]

  images = []
  for filename in filenames:
    raw = rawpy.imread(filename)
    im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    #im = im/1.0               #Change the factor to adjust brightness
    print(filename)
    images.append(im)
  return images

if __name__ == '__main__':
  
  # Read images
  print("Reading images ... ")
  
  # Read example images
  images = readImagesAndTimes()
  # Can't Align input images, so skip that step
  
  # Merge using Exposure Fusion
  print("Merging using Exposure Fusion ... ");
  mergeMertens = cv2.createMergeMertens()
  exposureFusion = mergeMertens.process(images)
  
  # Convert gt_full to 16 bit unsigned integers.
  # ptp means the value range from min to max 
  z = (65535*((exposureFusion - exposureFusion.min())/exposureFusion.ptp())).astype(np.uint16)
    
  # Save output image
  print("Saving output...")
  with open('exposure-fusion.png', 'wb') as f:
    writer = png.Writer(width=z.shape[1], height=z.shape[0], bitdepth=16)
    # Convert z to the Python list of lists expected by
    # the png writer.
    z2list = z.reshape(-1, z.shape[1]*z.shape[2]).tolist()
    writer.write(f, z2list)
  

