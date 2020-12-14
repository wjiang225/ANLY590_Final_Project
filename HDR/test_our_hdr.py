#uniform content loss + adaptive threshold + per_class_input + recursive G
#improvement upon cqf37
from __future__ import division
import os,time,scipy.io
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pdb
import rawpy
import glob
import cv2
import scipy.misc
from tensorflow.python.client import device_lib
	
#os.environ['CUDA_VISIBLE_DEVICES']='0'						#set cuda device to GPU	
#print (device_lib.list_local_devices())

input_dir = "short/"								        #short dir
gt_dir = 'long/'                  				        #long(ground_truth) dir
checkpoint_dir = 'checkpoint/'							#model dir
result_dir = 'result/'								#result dir

#get train and test IDs
train_fns = glob.glob(gt_dir + '0*.hdr')    				#train file names, starts with 0, ends with .png
train_ids = []
for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i])
    train_ids.append(int(train_fn[0:5]))

test_fns = glob.glob(gt_dir + '/0*.hdr')    				#test file names, starts with 1, ends with .png
test_ids = []
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append(int(test_fn[0:5]))

print(test_ids)

ps = 512 #patch size for training
save_freq = 500

DEBUG = 0
if DEBUG == 1:
  save_freq = 2
  train_ids = train_ids[0:5]
  test_ids = test_ids[0:5]


def lrelu(x):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels):

    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal( [pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )

    deconv_output =  tf.concat([deconv, x2],3)
    deconv_output.set_shape([None, None, None, output_channels*2])

    return deconv_output

def network(input):
    #Xavier initializer
	initializer = tf.initializers.glorot_uniform()
	filter0 = tf.Variable(initializer([3,3,4,32]))
	b0 = tf.Variable(tf.zeros([32]))
	conv1 = tf.nn.conv2d(input, filter0,[1,1,1,1],padding='SAME')
	conv1 = tf.nn.bias_add(conv1,b0)
	conv1 = lrelu(conv1)
	filter1 = tf.Variable(initializer([3,3,32,32]))
	b1 = tf.Variable(tf.zeros([32]))
	conv1 = tf.nn.conv2d(conv1, filter1,[1,1,1,1],padding='SAME')
	conv1 = tf.nn.bias_add(conv1,b1)
	conv1 = lrelu(conv1)
	pool1=tf.nn.max_pool(conv1,[1,2,2,1],[1,2,2,1],padding='SAME')
	
	filter2 = tf.Variable(initializer([3,3,32,64]))
	b2 = tf.Variable(tf.zeros([64]))
	conv2 = tf.nn.conv2d(pool1, filter2,[1,1,1,1],padding='SAME')
	conv2 = tf.nn.bias_add(conv2,b2)
	conv2 = lrelu(conv2)
	filter3 = tf.Variable(initializer([3,3,64,64]))
	b3 = tf.Variable(tf.zeros([64]))
	conv2 = tf.nn.conv2d(conv2, filter3,[1,1,1,1],padding='SAME')
	conv2 = tf.nn.bias_add(conv2,b3)
	conv2 = lrelu(conv2)
	pool2=tf.nn.max_pool(conv2,[1,2,2,1],[1,2,2,1],padding='SAME')
	
	filter4 = tf.Variable(initializer([3,3,64,128]))
	b4 = tf.Variable(tf.zeros([128]))
	conv3 = tf.nn.conv2d(pool2, filter4,[1,1,1,1],padding='SAME')
	conv3 = tf.nn.bias_add(conv3,b4)
	conv3 = lrelu(conv3)
	filter5 = tf.Variable(initializer([3,3,128,128]))
	b5 = tf.Variable(tf.zeros([128]))
	conv3 = tf.nn.conv2d(conv3, filter5,[1,1,1,1],padding='SAME')
	conv3 = tf.nn.bias_add(conv3,b5)
	conv3 = lrelu(conv3)
	pool3=tf.nn.max_pool(conv3,[1,2,2,1],[1,2,2,1],padding='SAME')
	
	filter6 = tf.Variable(initializer([3,3,128,256]))
	b6 = tf.Variable(tf.zeros([256]))
	conv4 = tf.nn.conv2d(pool3, filter6,[1,1,1,1],padding='SAME')
	conv4 = tf.nn.bias_add(conv4,b6)
	conv4 = lrelu(conv4)
	filter7 = tf.Variable(initializer([3,3,256,256]))
	b7 = tf.Variable(tf.zeros([256]))
	conv4 = tf.nn.conv2d(conv4, filter7,[1,1,1,1],padding='SAME')
	conv4 = tf.nn.bias_add(conv4,b7)
	conv4 = lrelu(conv4)
	pool4=tf.nn.max_pool(conv4,[1,2,2,1],[1,2,2,1],padding='SAME')	
	
	filter8 = tf.Variable(initializer([3,3,256,512]))
	b8 = tf.Variable(tf.zeros([512]))
	conv5 = tf.nn.conv2d(pool4, filter8,[1,1,1,1],padding='SAME')
	conv5 = tf.nn.bias_add(conv5,b8)
	conv5 = lrelu(conv5)
	filter9 = tf.Variable(initializer([3,3,512,512]))
	b9 = tf.Variable(tf.zeros([512]))
	conv5 = tf.nn.conv2d(conv5, filter9,[1,1,1,1],padding='SAME')
	conv5 = tf.nn.bias_add(conv5,b9)
	conv5 = lrelu(conv5)
	
	up6 =  upsample_and_concat( conv5, conv4, 256, 512  ) #todo:bug
	filter10 = tf.Variable(initializer([3,3,512,256]))
	b10 = tf.Variable(tf.zeros([256]))
	conv6 = tf.nn.conv2d(up6, filter10,[1,1,1,1],padding='SAME')
	conv6 = tf.nn.bias_add(conv6,b10)
	conv6 = lrelu(conv6)
	filter11 = tf.Variable(initializer([3,3,256,256]))
	b11 = tf.Variable(tf.zeros([256]))
	conv6 = tf.nn.conv2d(conv6, filter11,[1,1,1,1],padding='SAME')
	conv6 = tf.nn.bias_add(conv6,b11)
	conv6 = lrelu(conv6)
	
	up7 =  upsample_and_concat( conv6, conv3, 128, 256  )
	filter12 = tf.Variable(initializer([3,3,256,128]))
	b12 = tf.Variable(tf.zeros([128]))
	conv7 = tf.nn.conv2d(up7, filter12,[1,1,1,1],padding='SAME')
	conv7 = tf.nn.bias_add(conv7,b12)
	conv7 = lrelu(conv7)
	filter13 = tf.Variable(initializer([3,3,128,128]))
	b13 = tf.Variable(tf.zeros([128]))
	conv7 = tf.nn.conv2d(conv7, filter13,[1,1,1,1],padding='SAME')
	conv7 = tf.nn.bias_add(conv7,b13)
	conv7 = lrelu(conv7)

	up8 =  upsample_and_concat( conv7, conv2, 64, 128 )
	filter14 = tf.Variable(initializer([3,3,128,64]))
	b14 = tf.Variable(tf.zeros([64]))
	conv8 = tf.nn.conv2d(up8, filter14,[1,1,1,1],padding='SAME')
	conv8 = tf.nn.bias_add(conv8,b14)
	conv8 = lrelu(conv8)
	filter15 = tf.Variable(initializer([3,3,64,64]))
	b15 = tf.Variable(tf.zeros([64]))
	conv8 = tf.nn.conv2d(conv8, filter15,[1,1,1,1],padding='SAME')
	conv8 = tf.nn.bias_add(conv8,b15)
	conv8 = lrelu(conv8)

	up9 =  upsample_and_concat( conv8, conv1, 32, 64 )
	filter16 = tf.Variable(initializer([3,3,64,32]))
	b16 = tf.Variable(tf.zeros([32]))
	conv9 = tf.nn.conv2d(up9, filter16,[1,1,1,1],padding='SAME')
	conv9 = tf.nn.bias_add(conv9,b16)
	conv9 = lrelu(conv9)
	filter17 = tf.Variable(initializer([3,3,32,32]))
	b17 = tf.Variable(tf.zeros([32]))
	conv9 = tf.nn.conv2d(conv9, filter17,[1,1,1,1],padding='SAME')
	conv9 = tf.nn.bias_add(conv9,b17)
	conv9 = lrelu(conv9)
	
	filter18 = tf.Variable(initializer([1,1,32,12]))
	b18 = tf.Variable(tf.zeros([12]))
	conv10 = tf.nn.conv2d(conv9, filter18,[1,1,1,1],padding='SAME')
	conv10 = tf.nn.bias_add(conv10,b18)
	
	out = tf.depth_to_space(conv10,2)
	
	return out


def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32) 
    im = np.maximum(im - 512,0)/ (16383 - 512) 				#subtract the black level

    im = np.expand_dims(im,axis=2) 
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

sess=tf.Session()
in_image=tf.placeholder(tf.float32,[None,None,None,4])
gt_image=tf.placeholder(tf.float32,[None,None,None,3])
out_image=network(in_image)

saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
    
    
if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')

for test_id in test_ids:
    #test the first image in each sequence
    in_files = glob.glob(input_dir + '%05d_00*.ARW'%test_id)
    for k in range(len(in_files)):
        in_path = in_files[k]
        _, in_fn = os.path.split(in_path)
        print(in_fn)
        gt_files = glob.glob(gt_dir + '%05d_00*.hdr'%test_id)
        gt_path = gt_files[0]
        _, gt_fn = os.path.split(gt_path)
        in_exposure =  float(in_fn[9:-5])					#short exposure time(acquired from file names)
        gt_exposure =  float(gt_fn[9:-5])					#long exposure time(acquired from file names)
        ratio = (int)(min(gt_exposure/in_exposure,300))		#calculate exposure ratio 

        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw),axis=0) *ratio
        
        #im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
		
        input_full = np.minimum(input_full,1.0)
        
        output =sess.run(out_image,feed_dict={in_image: input_full})
        output = np.minimum(np.maximum(output,0),1)

        output = output[0,:,:,:]
        #print(output.shape)
        B,G,R = np.split(output, 3, axis = 2)
        #print(B.shape)
        output = np.concatenate([R, G, B], axis=2)
        
        #print(output.shape)
        
        scipy.misc.toimage(output*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + 'final/%5d_00_%d_out.jpg'%(test_id,ratio))








