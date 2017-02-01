#!/usr/bin/env python

# ros images
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2

import rospy


import sys, os, os.path, numpy as np
import time
import cv2


bridge = CvBridge()

label_img = cv2.imread("/home/xi/workspace/labels/slope/slope_11_image.png")
label_img_grey = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("label_img", label_img_grey)
cv2.waitKey(10)

def callback(data):
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
        bag_img_grey = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("bag_img", bag_img_grey)
        cv2.waitKey(10)

        mat_diff = abs(bag_img_grey - label_img_grey)
        cv2.imshow("diff_img", mat_diff)
        not_match = cv2.countNonZero(mat_diff)
        if not_match < 400000:
            print data.header.stamp

    except CvBridgeError as e:
        print(e)
	

def main(args):
	global cv_image
	rospy.init_node('find_label_match', anonymous=True)
	image_sub = rospy.Subscriber("/kinect2/qhd/image_color", Image, callback, queue_size=100)

	rospy.spin()
	# rate = rospy.Rate(2)   ## 2 means read image for every 0.5 second

	# while not rospy.is_shutdown():
		# msg = rospy.wait_for_message("/kinect2/qhd/image_color", Image)
		# callback(msg)
		# rate.sleep()		

	# cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)

