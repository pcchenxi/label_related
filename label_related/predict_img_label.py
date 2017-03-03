from __future__ import print_function, division, absolute_import

import tensorflow as tf
import numpy as np
import pandas as pd
import scipy as sp
import rospy
import time
import sys
import os
import cv2

import pickle

from skimage import io, transform
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# from colorize import colorize
# from class_mean_iou import class_mean_iou

sys.path.append('/home/fabian/segmentation/')
sys.path.append('/home/fabian/segmentation//datasets/')
sys.path.append('/home/fabian/segmentation//models')

import layers
import fcn8s
import util
import cityscapes


def callback(image):
#     global count, file_count
    
#     if count%300 != 0:
#         count += 1
#         return
#     path = '/home/xi/workspace/labels/rough/rough_' + str(file_count)
#     image = bridge.imgmsg_to_cv2(message)
#     cv2.imwrite(path + '_image.jpg', image)
    image = sp.misc.imresize(image, image_shape[1:], interp='bilinear')
    image_publisher.publish(bridge.cv2_to_imgmsg(image))
    image = image[..., ::-1] # bgr to rgb
    image = (image - image.mean()) / image.std()
    
    feed_dict = {image_op: image[np.newaxis, ...]}
    
    prediction = sess.run(predictions_op, feed_dict=feed_dict)
    prediction
    pickle.dump(prediction, "/home/xi/workspace/labels/slope/save.p", "wb")
#     prediction = colorize(prediction, cityscapes.augmented_labels)
#     cv2.imwrite(path + '_prediction.png', prediction)
    
#     prediction = prediction[..., ::-1


checkpoint, checkpoint_path, model_name = util.get_checkpoint('/home/fabian/tf_models/fcn8s_augment_finetune/')
print(checkpoint, checkpoint_path, model_name)
sess = tf.InteractiveSession()

image_shape = [1, 256, 512, 3]
image_op = tf.placeholder(tf.float32, shape=image_shape)

logits_op = fcn8s.inference(image_op)
# predictions_op = layers.predictions(logits_op)
predictions_op = tf.nn.softmax(logits_op)

init_op = tf.global_variables_initializer()
sess.run(init_op)

saver = tf.train.Saver()
saver.restore(sess, checkpoint)

img = cv2.imread('/home/xi/workspace/labels/slope/slope_1s_image.png')
callback(img)

