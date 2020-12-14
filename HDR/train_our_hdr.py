#uniform content loss + adaptive threshold + per_class_input + recursive G
#improvement upon cqf37
from __future__ import division
import os,time,scipy.io
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pdb
import rawpy														#raw processing library
import glob															#get file names from file system
import cv2															#read 16-bit png
from tensorflow.python.client import device_lib

input_dir = 'short/'										#short dir
gt_dir = 'long/'											#long(ground_truth) dir
checkpoint_dir = 'checkpoint/'									#model dir
result_dir = 'result/'										#result dir(for result preveiw)

#get train and test IDs
train_fns = glob.glob(gt_dir + '0*.hdr')							#train file names, starts with 0, ends with .png
train_ids = []
for i in range(len(train_fns)):
	_, train_fn = os.path.split(train_fns[i])
	train_ids.append(int(train_fn[0:5]))							#split ids from file names 

test_fns = glob.glob(gt_dir + '/1*.ARW')							#test file names, starts with 1, ends with .ARW(sony RAW)
test_ids = []
for i in range(len(test_fns)):
	_, test_fn = os.path.split(test_fns[i])							#split ids from file names 
	test_ids.append(int(test_fn[0:5]))

ps = 512 															#patch size for training
save_freq = 500
model_freq = 100													#save every 100ep
lastepoch = 2201

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
	deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1])					 
	deconv_output =  tf.concat([deconv, x2],3)					    # 再把两个(,,,256)拼在一起变成(,,,512)
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
	
def pack_hdr(bt_stream):
	hdr = np.frombuffer(bt_stream, dtype=np.uint8)
	hdr = np.expand_dims(hdr,axis=1)
	r = hdr[0:48484352:4,:]
	g = hdr[1:48484352:4,:]
	b = hdr[2:48484352:4,:]
	e = hdr[3:48484352:4,:]
	r = np.reshape(r,(2848,4256,1))
	g = np.reshape(g,(2848,4256,1))
	b = np.reshape(b,(2848,4256,1))
	e = np.reshape(e,(2848,4256,1))
	
	out = np.concatenate((r, g, b), axis=2)
	#out = np.concatenate((r, g, b, e), axis=2)
	return out
	
def pack_raw(raw):
	#pack Bayer image to 4 channels
	im = raw.raw_image_visible.astype(np.float32) 
	im = np.maximum(im - 512,0)/ (16383 - 512) 						#subtract the black level

	im = np.expand_dims(im,axis=2) 
	img_shape = im.shape
	H = img_shape[0]
	W = img_shape[1]

	out = np.concatenate((im[0:H:2,0:W:2,:], 
					   im[0:H:2,1:W:2,:],
					   im[1:H:2,1:W:2,:],
					   im[1:H:2,0:W:2,:]), axis=2)
	return out



#GPU options
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

sess=tf.Session()

in_image=tf.placeholder(tf.float32,[None,None,None,4])
gt_image=tf.placeholder(tf.float32,[None,None,None,3])				
out_image=network(in_image)

G_loss=tf.reduce_mean(tf.abs(out_image - gt_image))			 #loss function, maybe improvement, SSIM
#ssim = tf.image.ssim(im1, im2, max_val=1.0)

t_vars=tf.trainable_variables()
lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())

ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
	print('loaded '+ckpt.model_checkpoint_path)
	saver.restore(sess,ckpt.model_checkpoint_path)

#Raw data takes long time to load. Keep them in memory after loaded.
gt_images=[None]*6000
input_images = {}
input_images['16'] = [None]*len(train_ids)				  		#divide into classes according to exposure ratio, fixed  
#input_images['250'] = [None]*len(train_ids)					#there were 3 classed in SID code, but in our dataset,
#input_images['100'] = [None]*len(train_ids)					#the exposure ratio is fixed, 8 or 16, depends on which dataset 
																#used, minus3 or short(minus4)
g_loss = np.zeros((5000,1))

allfolders = glob.glob('./result/*0')

for folder in allfolders:
	lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-4
for epoch in range(lastepoch,4001):
	if os.path.isdir("result/%04d"%epoch):
		continue	
	cnt=0
	if epoch > 2000:
		learning_rate = 1e-5 
  
	for ind in np.random.permutation(len(train_ids)):
		# get the path from image id
		train_id = train_ids[ind]
		in_files = glob.glob(input_dir + '%05d_00*.ARW'%train_id)
		in_path = in_files[np.random.random_integers(0,len(in_files)-1)]
		_, in_fn = os.path.split(in_path)

		gt_files = glob.glob(gt_dir + '%05d_00*.hdr'%train_id)		
		gt_path = gt_files[0]
		_, gt_fn = os.path.split(gt_path)
		in_exposure =  float(in_fn[9:-5])
		gt_exposure =  float(gt_fn[9:-5])
		ratio = min(gt_exposure/in_exposure,300)					#calculate exposure ratio
		st=time.time()
		cnt+=1

		if input_images[str(ratio)[0:2]][ind] is None:
			raw = rawpy.imread(in_path)
			input_images[str(ratio)[0:2]][ind] = np.expand_dims(pack_raw(raw),axis=0) *ratio

			print(gt_path)
			'''
			# method 1: read raw hdr file in rgbe format
			gt_file = open(gt_path, 'rb')
			header = gt_file.read(78)
			gt_bytes = gt_file.read()
			gt_file.close()
			gt_images[ind] = np.expand_dims(np.float32(pack_hdr(gt_bytes)/255.0),axis = 0)# 8-bit, 4 channels
			'''
			# method 2: use Reinhard tonemapped image as gt
			hdr = cv2.imread(gt_path,-1)
			tonemapReinhard = cv2.createTonemapReinhard(3.0,-2.5,0,0)
			gt_images[ind] = np.expand_dims(tonemapReinhard.process(hdr),axis = 0)
		#crop
		H = input_images[str(ratio)[0:2]][ind].shape[1]
		W = input_images[str(ratio)[0:2]][ind].shape[2]

		xx = np.random.randint(0,W-ps)
		yy = np.random.randint(0,H-ps)
		input_patch = input_images[str(ratio)[0:2]][ind][:,yy:yy+ps,xx:xx+ps,:]
		gt_patch = gt_images[ind][:,yy*2:yy*2+ps*2,xx*2:xx*2+ps*2,:]
		
		if np.random.randint(2,size=1)[0] == 1:  # random flip 
			input_patch = np.flip(input_patch, axis=1)
			gt_patch = np.flip(gt_patch, axis=1)
		if np.random.randint(2,size=1)[0] == 1: 
			input_patch = np.flip(input_patch, axis=0)
			gt_patch = np.flip(gt_patch, axis=0)
		if np.random.randint(2,size=1)[0] == 1:  # random transpose 
			input_patch = np.transpose(input_patch, (0,2,1,3))
			gt_patch = np.transpose(gt_patch, (0,2,1,3))
		
		input_patch = np.minimum(input_patch,1.0)
		
		_,G_current,output=sess.run([G_opt,G_loss,out_image],feed_dict={in_image:input_patch,gt_image:gt_patch,lr:learning_rate})
		output = np.minimum(np.maximum(output,0),1)
		g_loss[ind]=G_current

		print("%d %d Loss=%.3f Time=%.3f"%(epoch,cnt,np.mean(g_loss[np.where(g_loss)]),time.time()-st))

		if cnt == 1:
			with open('LossRecord.txt','a') as file:
				write_str = '%d %f\n'%(epoch,np.mean(g_loss[np.where(g_loss)]))
				file.write(write_str)
		
		if epoch%save_freq==0:
		  if not os.path.isdir(result_dir + '%04d'%epoch):
			  os.makedirs(result_dir + '%04d'%epoch)
		  
		  temp = np.concatenate((gt_patch[0,:,:,:],output[0,:,:,:]),axis=1)
		  scipy.misc.toimage(temp*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%04d/%05d_00_train_%d.jpg'%(epoch,train_id,ratio))
	
	if epoch%model_freq==0:
		saver.save(sess, checkpoint_dir + 'model.ckpt')
	if epoch%save_freq==0:
		saver.save(sess, checkpoint_dir + f'{str(epoch)}/'+'model.ckpt')
		
		
		
		
		