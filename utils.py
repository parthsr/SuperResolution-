"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""
import time
import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

from PIL import Image  # for loading images as YCbCr format
import scipy.misc
import scipy.ndimage
import numpy as np

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def read_data(path):
  """
  Read h5 format data file
  
  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label

def preprocess(path,i,j,scale=3):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation

  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  
  image = imread(path, is_grayscale=True)#to grayscale -- training image
  label_ = modcrop(image, scale)#creating a label from the image by reducing the height and the breadth of the image by the scale size
  # Must be normalized
  image = image / 255.
  label_ = label_ / 255.
  input_ = scipy.ndimage.interpolation.zoom(label_, (1./scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input_, (scale/1.), prefilter=False)#this zoom in and zoom out reduces the clarity of the image

  if not os.path.exists("Training&Test"):
    os.makedirs("Training&Test")
    plt.imshow(input_) 
    plt.savefig('Training&Test/input'+str(i)+'.png')
    plt.imshow(label_)
    plt.savefig('Training&Test/labels'+str(i)+'.png')
  return input_, label_ #returned input and labels both of the same shape

def prepare_data(sess, dataset):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
  if FLAGS.is_train:
    filenames = os.listdir(dataset)#list of all directories in train folder
    data_dir = os.path.join(os.getcwd(), dataset)#training data folder
    data = glob.glob(os.path.join(data_dir, "*.bmp"))#returns a string containing all the *.bmp names in the files
  else:
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
    data = glob.glob(os.path.join(data_dir, "*.bmp"))

  return data

def make_data(sess, data, label):
  """
  Make input data as h5 file format
  Depending on 'is_train' (flag value), savepath would be changed.
  """
  if FLAGS.is_train:
    savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
  else:
    savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)

def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
  '''modcrop is done so that when zoom in and out is done using the scale float variable,
  no remainder is left. The height and the width should be divisible by the scaling factor'''



  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def input_setup(sess, config,i):
  """
  Read image files and make their sub-images and saved them as a h5 file format.
  """
  # Load data path
  if config.is_train:
    data = prepare_data(sess, dataset="Train")#contains all the training pics names
  else:
    data = prepare_data(sess, dataset="Test")

  sub_input_sequence = []#contains all 33 x 33 subsets of a single image for all images
  sub_label_sequence = []#contains all 21 x 21 subsets of single label for all labels. These labels are a part of input image 31 x 31
  padding = abs(config.image_size - config.label_size) / 2 # 6
  if config.is_train:
    for i in xrange(len(data)):
      input_, label_ = preprocess(data[i],i, config.scale)#sending each name of the pic

      if len(input_.shape) == 3:
        h, w, _ = input_.shape
      else:
        h, w = input_.shape

      for x in range(0, h-config.image_size+1, config.stride):
        '''this is where the meaning of stride is known to us. Here stride is used to make the subsets. SO the move is from 0 to 0+stride over the matrix of image'''
        for y in range(0, w-config.image_size+1, config.stride):
          sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]#CAN BE LESSER THAN 33 X 33 ACCORDING TO THE SIZE OF THE IMAGE LEFT
          '''so a subset of those images are stored in sub_image'''
          sub_label = label_[x+padding:x+padding+config.label_size, y+padding:y+padding+config.label_size] # [21 x 21]
          
          # Make channel value
          sub_input = sub_input.reshape([config.image_size, config.image_size, 1])  
          sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

          sub_input_sequence.append(sub_input)
          sub_label_sequence.append(sub_label)

  else:
    input_, label_ = preprocess(data[i],0, config.scale)

    image_path = os.path.join(os.getcwd(), config.sample_dir)
    image_path = os.path.join(image_path, "test_image.png")
    plt.imshow(input_)
    plt.savefig(image_path)
    image_path = os.path.join(os.getcwd(), config.sample_dir)
    image_path = os.path.join(image_path, "test_label.png")
    plt.imshow(label_)
    plt.savefig(image_path)

    if len(input_.shape) == 3:
      h, w, _ = input_.shape
    else:
      h, w = input_.shape

    # Numbers of sub-images in height and width of image are needed to compute merge operation.
    nx = ny = 0 
    for x in range(0, h-config.image_size+1, config.stride):
      nx += 1; ny = 0
      for y in range(0, w-config.image_size+1, config.stride):
        ny += 1
        sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
        sub_label = label_[x+padding:x+padding+config.label_size, y+padding:y+padding+config.label_size] # [21 x 21]
        
        sub_input = sub_input.reshape([config.image_size, config.image_size, 1])  
        sub_label = sub_label.reshape([config.label_size, config.label_size, 1])
        sub_input_sequence.append(sub_input)
        sub_label_sequence.append(sub_label)

  """
  len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
  (sub_input_sequence[0]).shape : (33, 33, 1)
  """
  # Make list to numpy array. With this transform
  arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
  arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]

  make_data(sess, arrdata, arrlabel)

  if not config.is_train:
    return nx, ny
    
def imsave(image, path):
  return scipy.misc.imsave(path, image)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h*size[0], w*size[1], 1))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img
