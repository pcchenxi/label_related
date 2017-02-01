import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import tree
from sklearn.externals import joblib
import cv2
import numpy as np


file_num = 11
rosbag_name = 'hogwarts'
base_path_gt         = '/home/xi/workspace/labels/' + rosbag_name + '/' + rosbag_name + '_'  
base_path_geo_result = '/home/xi/workspace/labels/output/' + rosbag_name + '/' + rosbag_name + '_'

result = np.zeros([4,4], dtype=np.int)

for i in range(2, file_num):
    path_gt        = base_path_gt + str(i) + '_label.png'
    path_g_label   = base_path_geo_result + str(i) + '_label.png'
    path_v_label   = base_path_geo_result + str(i) + '_vision.png'

    print path_gt, path_g_label, path_v_label
    label_gt = cv2.imread(path_gt, 0)
    label_geo = cv2.imread(path_g_label, 0)
    
    rows, cols = label_geo.shape
    # print rows, cols
    # result = np.zeros([4,4], dtype=np.int)
    x = []
    y = []

    for row in xrange(rows):
        for col in xrange(cols):
            gt = label_gt[row, col]/50
            geo = label_geo[row, col]/50
            # print gt, geo, label_geo[row, col]
            if gt < 1 or gt > 3 or geo < 1 or geo > 3:
                continue

            result[gt, geo] = result[gt, geo] + 1

print result
    # cv2.imshow("ground_truth", gt)
    # cv2.imshow("geo_result", geo_result)
    # cv2.waitKey(0)
