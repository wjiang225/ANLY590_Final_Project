import cv2
import glob
import numpy as np

gt_dir = './long/'
result_dir = './long_jpg/'
fns = glob.glob(gt_dir + '/0*.hdr')


for fn in fns:
    name = result_dir + fn.split('\\')[1].split('.hdr')[0] + '.jpg'
    print(name)
    hdr = cv2.imread(fn,-1)
    tonemapReinhard = cv2.createTonemapReinhard(3.0,-2.5,0,0)#本来是-2.5
    cv2.imwrite(name, tonemapReinhard.process(hdr)*255)