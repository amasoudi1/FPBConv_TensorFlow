# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import scipy.io as sc
import sys
#import scipy as sc
#import mat4py as mt
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(0)

def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(s): return tf.cast(tf.greater(s,0),dtype=tf.float32)
def tf_softmax(x): return tf.nn.softmax(x)
def np_sigmoid(x): 1/(1 + np.exp(-1 *x))

# --- make class ---
class conlayer_left():
    
    def __init__(self,ker,in_c,out_c):
        self.w = tf.Variable(tf.random_normal([ker,ker,in_c,out_c],stddev=0.05))
        #self.b = tf.Variable(tf.zeros[out_c])
    def feedforward(self,input,stride=1,dilate=1):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides = [1,stride,stride,1],padding='SAME')
        #self.layerB = tf.nn_add(self.layer, self.b)
        self.layer = tf_relu(self.layer)
        return self.layer

class conlayer_right():
    
    def __init__(self,ker,in_c,out_c):
        self.w = tf.Variable(tf.random_normal([ker,ker,out_c,in_c],stddev=0.05))
    def feedforward(self,input,stride=1,dilate=1,output=1):
        self.input  = input

        current_shape_size = input.shape

        self.layer = tf.nn.conv2d_transpose(input,self.w,
                                           output_shape=[batch_size] + [int(current_shape_size[1].value*2),int(current_shape_size[2].value*2),int(current_shape_size[3].value/2)],strides=[1,2,2,1],padding='SAME')
        self.layerA = tf_relu(self.layer)
        return self.layerA

# --- get data ---
data_location = "./train/recon/"
train_data = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".mat" in filename.lower():  # check whether the file's DICOM
            train_data.append(os.path.join(dirName,filename))

data_location = "./train/grount/"
train_data_gt = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".mat" in filename.lower():  # check whether the file's DICOM
            train_data_gt.append(os.path.join(dirName,filename))


train_images = np.zeros(shape=(400,64,64,1))
train_labels = np.zeros(shape=(400,64,64,1))

for file_index in range(len(train_data)):
    data_temp = sc.loadmat(train_data[file_index])
    train_images[file_index,:,:]   = np.expand_dims(np.array(data_temp['fi']),axis=2)
    data_temp = sc.loadmat(train_data_gt[file_index])
    train_labels[file_index,:,:]   = np.expand_dims(np.array(data_temp['ti']),axis=2)

train_images = 1000*(train_images - train_images.min()) / (train_images.max() - train_images.min()) - 500
train_labels = 1000*(train_labels - train_labels.min()) / (train_labels.max() - train_labels.min()) - 500

# --- hyper ---
num_epoch = 1000
init_lr = 0.01
batch_size = 4

# --- make layer ---
# left
l1_1 = conlayer_left(3,1,16)
l1_2 = conlayer_left(3,16,16)
l1_3 = conlayer_left(3,16,16)

l2_1 = conlayer_left(3,16,32)
l2_2 = conlayer_left(3,32,32)

l3_1 = conlayer_left(3,32,64)
l3_2 = conlayer_left(3,64,64)

l4_1 = conlayer_left(3,64,128)
l4_2 = conlayer_left(3,128,128)

l5_1 = conlayer_left(3,128,256)
l5_2 = conlayer_left(3,256,256)

# right
l6_1 = conlayer_right(3,256,128)
l6_2 = conlayer_left(3,256,128)
l6_3 = conlayer_left(3,128,128)

l7_1 = conlayer_right(3,128,64)
l7_2 = conlayer_left(3,128,64)
l7_3 = conlayer_left(3,64,64)

l8_1 = conlayer_right(3,64,32)
l8_2 = conlayer_left(3,64,32)
l8_3 = conlayer_left(3,32,32)

l9_1 = conlayer_right(3,32,16)
l9_2 = conlayer_left(3,32,16)
l9_3 = conlayer_left(3,16,16)

l10_1 = conlayer_left(3,16,1)

l11_final = conlayer_left(3,1,1)

# ---- make graph ----
x = tf.placeholder(shape=[None,64,64,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,64,64,1],dtype=tf.float32)
is_training = tf.placeholder(tf.bool)

layer1_1 = l1_1.feedforward(x)
layer1_2 = l1_2.feedforward(tf.layers.batch_normalization(
    inputs=layer1_1,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))
layer1_3 = l1_3.feedforward(tf.layers.batch_normalization(
    inputs=layer1_2,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))

layer2_Input = tf.nn.max_pool(layer1_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer2_1 = l2_1.feedforward(tf.layers.batch_normalization(
    inputs=layer2_Input,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))
layer2_2 = l2_2.feedforward(tf.layers.batch_normalization(
    inputs=layer2_1,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))

layer3_Input = tf.nn.max_pool(layer2_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer3_1 = l3_1.feedforward(tf.layers.batch_normalization(
    inputs=layer3_Input,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))
layer3_2 = l3_2.feedforward(tf.layers.batch_normalization(
    inputs=layer3_1,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))


layer4_Input = tf.nn.max_pool(layer3_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer4_1 = l4_1.feedforward(tf.layers.batch_normalization(
    inputs=layer4_Input,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))
layer4_2 = l4_2.feedforward(tf.layers.batch_normalization(
    inputs=layer4_1,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))

layer5_Input = tf.nn.max_pool(layer4_2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer5_1 = l5_1.feedforward(tf.layers.batch_normalization(
    inputs=layer5_Input,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))
layer5_2 = l5_2.feedforward(tf.layers.batch_normalization(
    inputs=layer5_1,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))


layer6_1 = l6_1.feedforward(tf.layers.batch_normalization(
    inputs=layer5_2,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))
layer6_Input = tf.concat([layer4_2,layer6_1],axis=3)
layer6_2 = l6_2.feedforward(tf.layers.batch_normalization(
    inputs=layer6_Input,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))
layer6_3 = l6_3.feedforward(tf.layers.batch_normalization(
    inputs=layer6_2,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))


layer7_1 = l7_1.feedforward(tf.layers.batch_normalization(
    inputs=layer6_3,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))
layer7_Input = tf.concat([layer3_2,layer7_1],axis=3)
layer7_2 = l7_2.feedforward(tf.layers.batch_normalization(
    inputs=layer7_Input,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))
layer7_3 = l7_3.feedforward(tf.layers.batch_normalization(
    inputs=layer7_2,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))

layer8_1 = l8_1.feedforward(tf.layers.batch_normalization(
    inputs=layer7_3,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))
layer8_Input = tf.concat([layer2_2,layer8_1],axis=3)
layer8_2 = l8_2.feedforward(tf.layers.batch_normalization(
    inputs=layer8_Input,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))
layer8_3 = l8_3.feedforward(tf.layers.batch_normalization(
    inputs=layer8_2,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))

layer9_1 = l9_1.feedforward(tf.layers.batch_normalization(
    inputs=layer8_3,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))
layer9_Input = tf.concat([layer1_3,layer9_1],axis=3)
layer9_2 = l9_2.feedforward(tf.layers.batch_normalization(
    inputs=layer9_Input,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))
layer9_3 = l9_3.feedforward(tf.layers.batch_normalization(
    inputs=layer9_2,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))

layer10 = l10_1.feedforward(tf.layers.batch_normalization(
    inputs=layer9_3,
    axis=-1,
    momentum=0.9,
    epsilon=0.001,
    center=True,
    scale=True,
    training = is_training))

layer11 = l11_final.feedforward(x + layer10)


#update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#with tf.control_dependencies(update_ops):
#    train_op = auto_train.minimize(cost)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
cost = tf.reduce_mean(tf.square(layer11 - y))
with tf.control_dependencies(update_ops):
    auto_train = tf.train.GradientDescentOptimizer(learning_rate=init_lr).minimize(cost)


# --- start session ---
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for iter in range(num_epoch):

        # train
        for current_batch_index in range(0,len(train_images),batch_size):
            current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
            current_label = train_labels[current_batch_index:current_batch_index+batch_size,:,:,:]
            sess_results = sess.run([cost,auto_train],feed_dict={x:current_batch,y:current_label,is_training:True})
            print(' Iter: ', iter, " Cost:  %.32f"% sess_results[0],end='\r')
        print('\n-----------------------')
        train_images,train_labels = shuffle(train_images,train_labels)

        # if iter % 2 == 0:
        #     test_example =   train_images[:2,:,:,:]
        #     test_example_gt = train_labels[:2,:,:,:]
        #     sess_results = sess.run([layer10],feed_dict={x:test_example})
        #
        #     sess_results = sess_results[0][0,:,:,:]
        #     test_example = test_example[0,:,:,:]
        #     test_example_gt = test_example_gt[0,:,:,:]
        #
        #     plt.figure()
        #     plt.imshow(np.squeeze(test_example),cmap='gray')
        #     plt.axis('off')
        #     plt.title('Original Image')
        #     plt.savefig('train_change/'+str(iter)+"a_Original_Image.png")
        #
        #     plt.figure()
        #     plt.imshow(np.squeeze(test_example_gt),cmap='gray')
        #     plt.axis('off')
        #     plt.title('Ground Truth Mask')
        #     plt.savefig('train_change/'+str(iter)+"b_Original_Mask.png")
        #
        #     plt.figure()
        #     plt.imshow(np.squeeze(sess_results),cmap='gray')
        #     plt.axis('off')
        #     plt.title('Generated Mask')
        #     plt.savefig('train_change/'+str(iter)+"c_Generated_Mask.png")
        #
        #     plt.figure()
        #     plt.imshow(np.multiply(np.squeeze(test_example),np.squeeze(test_example_gt)),cmap='gray')
        #     plt.axis('off')
        #     plt.title("Ground Truth Overlay")
        #     plt.savefig('train_change/'+str(iter)+"d_Original_Image_Overlay.png")
        #
        #     plt.figure()
        #     plt.axis('off')
        #     plt.imshow(np.multiply(np.squeeze(test_example),np.squeeze(sess_results)),cmap='gray')
        #     plt.title("Generated Overlay")
        #     plt.savefig('train_change/'+str(iter)+"e_Generated_Image_Overlay.png")
        #
        #     plt.close('all')


    # for data_index in range(0,len(train_images),batch_size):
    #     current_batch = train_images[current_batch_index:current_batch_index+batch_size,:,:,:]
    #     current_label = train_labels[current_batch_index:current_batch_index+batch_size,:,:,:]
    #     sess_results = sess.run(layer10,feed_dict={x:current_batch})
    #
    #     plt.figure()
    #     plt.imshow(np.squeeze(current_batch[0,:,:,:]),cmap='gray')
    #     plt.axis('off')
    #     plt.title(str(data_index)+"a_Original Image")
    #     plt.savefig('gif/'+str(data_index)+"a_Original_Image.png")
    #
    #     plt.figure()
    #     plt.imshow(np.squeeze(current_label[0,:,:,:]),cmap='gray')
    #     plt.axis('off')
    #     plt.title(str(data_index)+"b_Original Mask")
    #     plt.savefig('gif/'+str(data_index)+"b_Original_Mask.png")
    #
    #     plt.figure()
    #     plt.imshow(np.squeeze(sess_results[0,:,:,:]),cmap='gray')
    #     plt.axis('off')
    #     plt.title(str(data_index)+"c_Generated Mask")
    #     plt.savefig('gif/'+str(data_index)+"c_Generated_Mask.png")
    #
    #     plt.figure()
    #     plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(current_label[0,:,:,:])),cmap='gray')
    #     plt.axis('off')
    #     plt.title(str(data_index)+"d_Original Image Overlay")
    #     plt.savefig('gif/'+str(data_index)+"d_Original_Image_Overlay.png")
    #
    #     plt.figure()
    #     plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(sess_results[0,:,:,:])),cmap='gray')
    #     plt.axis('off')
    #     plt.title(str(data_index)+"e_Generated Image Overlay")
    #     plt.savefig('gif/'+str(data_index)+"e_Generated_Image_Overlay.png")
    #
    #     plt.close('all')


# -- end code --