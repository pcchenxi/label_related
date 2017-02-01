import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import tree
import cv2


from sklearn.externals import joblib
# joblib.dump(clf, 'slope_1_model.pkl') 
clf = joblib.load('svm_slope_12_model.pkl') 
result = np.zeros([4,4], dtype=np.int)

img = cv2.imread("/home/xi/workspace/labels/slope/slope_1_label.png")
img_result_r = np.zeros(img.shape, dtype=np.uint8)
# img_result_w = np.zeros(img.shape, dtype=np.uint8)
# cv2.imshow("img", img)
# cv2.waitKey(10)
i = 0
with open("slope_4.txt") as f:
    content = f.readlines()
    x = []
    y = []
    test_num = len(content)
    correct_num = 0
    for line in content:
        # i=i+1
        # if i%1000 == 1:
        #     print i, test_num
        features = line.split(' ')
        
        features_clean = []
        for feature in features:
            feature = feature.replace('\n', '')
            features_clean.append(float(feature))
        # print features_clean

        x2 = []
        x_test = features_clean[1:5]
        x2.append(x_test)
        y_test = int(features_clean[0]/50)
        predict_label = clf.predict(x2)
        result[y_test, predict_label] = result[y_test, predict_label] + 1
        if y_test == predict_label:
            correct_num = correct_num+1
            # img_result_r[int(features_clean[5]), int(features_clean[6])] = [255, 255, 255]
            # img_result_w[int(features_clean[5]), int(features_clean[6])] = [255, 255, 255]


accuracy = float(correct_num)/float(test_num)
print correct_num, test_num, accuracy
# cv2.imshow("img_result_r", img_result_r)
# cv2.imshow("img_result_w", img_result_w)
# cv2.waitKey(0)

print result
