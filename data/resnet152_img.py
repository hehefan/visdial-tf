import os
import sys
import numpy as np
import h5py
import argparse
import json

import tensorflow as tf
from slim.nets import resnet_v1
from slim.preprocessing import vgg_preprocessing

parser = argparse.ArgumentParser(description='Preprocess image for Visual Dialogue')
parser.add_argument('--inputJson', type=str, default='visdial_params.json', help='Path to JSON file')
parser.add_argument('--imageRoot', type=str, default='images/', help='Path to COCO image root')
parser.add_argument('--cnnModel', type=str, default='resnet_v1_152.ckpt', help='Path to the CNN model')
parser.add_argument('--batchSize', type=int, default=50, help='Batch size')
parser.add_argument('--outName', type=str, default='resnet_img.h5', help='Output name')
parser.add_argument('--gpuid', type=str, default='3', help='Which gpu to use.')
parser.add_argument('--imgSize', type=int, default=224)

args = parser.parse_args()
slim = tf.contrib.slim

if True:
  os.system('wget http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz')
  os.system('tar -xvzf resnet_v1_152_2016_08_28.tar.gz')
  os.system('rm resnet_v1_152_2016_08_28.tar.gz')

def extract_feature(imgList, args):
  tf.reset_default_graph()

  queue = tf.train.string_input_producer(imgList, num_epochs=None, shuffle=False)
  reader = tf.WholeFileReader()

  img_path, img_data = reader.read(queue)
  img = vgg_preprocessing.preprocess_image(tf.image.decode_jpeg(contents=img_data, channels=3), args.imgSize, args.imgSize)
  img = tf.expand_dims(img, 0)
  with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    net, end_points = resnet_v1.resnet_v1_152(inputs=img, is_training=False)
  feat1 = end_points['resnet_v1_152/block4']
  feat2 = end_points['pool5']

  saver = tf.train.Saver()
  init_op = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init_op)
    saver.restore(sess, args.cnnModel)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    feats1 = []
    feats2 = []
    for i in range(len(imgList)):
      f1, f2 = sess.run([feat1, feat2]) # f1: (1, 7, 7, 2048)   f2: (1, 1, 1, 2048)
      feats1.append(f1[0])
      feats2.append(f2[0][0][0])
    coord.request_stop()
    coord.join(threads)
  return feats1, feats2

jsonFile = json.load(open(args.inputJson, 'r'))
trainList = []
for img in jsonFile['unique_img_train']:
  trainList.append(os.path.join(args.imageRoot, 'train2014/COCO_train2014_%012d.jpg'%img))
valList = []
for img in jsonFile['unique_img_val']:
  valList.append(os.path.join(args.imageRoot, 'val2014/COCO_val2014_%012d.jpg'%img))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

trainFeats7, trainFeats1 = extract_feature(trainList, args)
valFeats7, valFeats1 = extract_feature(valList, args)

print 'Saving hdf5...'
f = h5py.File(args.outName, 'w')
f.create_dataset('images_train_7', data=trainFeats7)
f.create_dataset('images_train_1', data=trainFeats1)
f.create_dataset('images_val_7', data=valFeats7)
f.create_dataset('images_val_1', data=valFeats1)
f.close()
