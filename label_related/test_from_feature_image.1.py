import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import tree
from sklearn.externals import joblib
import cv2
import numpy as np
from sklearn import preprocessing


def get_feature(rosbag_name, x, y):
    base_path_geo_result = '/home/xi/workspace/labels/output/' + rosbag_name + '/' + rosbag_name + '_'

    for i in range(1,12):   
        file_path = base_path_geo_result + str(i) + '_features.txt'
        print file_path
        with open(file_path) as f:
            content = f.readlines()
            for line in content:
                # print line
                features = line.split(' ')
                
                features_clean = []
                for feature in features:
                    feature = feature.replace('\n', '')
                    features_clean.append(float(feature))
                # print features_clean

                x_new = features_clean[1:5]
                y_new = int(features_clean[0])
                x.append(x_new)
                y.append(y_new)

    return x, y

x = []
y = []
sum = np.zeros([4,4], dtype=np.int)
result = np.zeros([4,4], dtype=np.int)
x, y = get_feature('snow_grass', x, y)
x, y = get_feature('hogwarts', x, y)

x_scaled = preprocessing.scale(x)

end = len(x_scaled)

clf = joblib.load('tree_hogwarts_123_model.pkl') 

for i in range(1, end):
    if i%10 == 0:
        continue

    x_test = []
    x_test.append(x_scaled[i])

    predict_label = clf.predict(x_test)

    if y[i] > 4 or predict_label > 4:
        continue 

    result[y[i], predict_label] = result[y[i], predict_label] + 1


sum_1 = result[1,1] + result[1,2] + result[1,3]
sum_2 = result[2,1] + result[2,2] + result[2,3]
sum_3 = result[3,1] + result[3,2] + result[3,3]

sum[0:1] = 1
sum[1:2] = sum_1
sum[2:3] = sum_2
sum[3:4] = sum_3
print sum

print result
result = result*100/sum;

print result
print "dome"


