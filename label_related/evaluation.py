import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import cv2
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt

## classifiers
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp

# names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
#          "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#          "Naive Bayes", "QDA"]

# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis()]


names = [
        # "Nearest Neighbors",
        # "Linear SVM", "RBF SVM",
         "Decision Tree", 
        #  "Random Forest", 
        #  "Neural Net", 
        #  "AdaBoost",
        #  "Naive Bayes"
         ]

classifiers = [
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # MLPClassifier(alpha=1),
    # AdaBoostClassifier(),
    # GaussianNB()
    ]


def normolize_data(data, feature_norms):
    for i in range(len(data[0])):
        feature = np.zeros( [len(data), 1], dtype=np.float32 )
        mean = feature_norms[i][0]
        std = feature_norms[i][1]
        for j in range(len(data)):
            data[j][i] = (data[j][i] - mean)/std

    return data


def voted_prediction(img, row, col, radius, ori_predict):
    start_row = row - radius/2
    end_row   = row + radius/2
    start_col = col - radius/2
    end_col   = col + radius/2

    img_rows, img_cols = img.shape

    if start_row < 0:
        start_row = 0
    if start_col < 0:
        start_col = 0

    if end_row > img_rows:
        end_row = img_rows
    if end_col > img_cols:
        end_col = img_cols        

    sub_img = img[start_row:end_row, start_col:end_col]
    hist,bins = np.histogram(sub_img.ravel(),6,[0,256])
    hist = hist[1:]
    new_predit = hist.argmax()+1

    return new_predit


def get_feature(rosbag_name, x, y, img_lengths, img_uvs, index_start = 2, index_end = 12):
    base_path_geo_result = '/home/xi/workspace/labels/output/' + rosbag_name + '/' + rosbag_name + '_'
    for i in range(index_start, index_end):   
        file_path = base_path_geo_result + str(i) + '_features.txt'
        # print file_path
        with open(file_path) as f:
            content = f.readlines()
            img_lengths.append(img_lengths[-1] + len(content))
            for line in content:
                # print line
                features = line.split(' ')
                
                features_clean = []
                for feature in features:
                    feature = feature.replace('\n', '')
                    features_clean.append(float(feature))
                # print features_clean

                if features_clean[0] < 1 or features_clean[0] > 3:
                    img_lengths[-1] -= 1 
                    continue

                # if features_clean[0] == 1 or features_clean[0] == 2:
                #     features_clean[0] = 1

                h_d = features_clean[2]
                s   = features_clean[3]
                r_1 = features_clean[4]
                r_2 = features_clean[5]

                if h_d > 0.2 or s > 0.4:
                    features_clean[0] = 3

                x_new = []
                x_new = features_clean[2:4]
                x_new.append(features_clean[5])
                # x_new.append(features_clean[6]) # vision

                y_new = int(features_clean[0]) # true label

                x.append(x_new)
                y.append(y_new)

                uv = features_clean
                img_uvs.append(uv)
                # img_uvs.append(i)


    return x, y, img_lengths, img_uvs

 
x = []
y = []
x_train = []
y_train = []
x_test = []
y_test = []


################################ get trainning and test set ############################

img_lengths_train = []
img_uvs_train = []
img_lengths_train.append(0)
# x_train, y_train, img_lengths_train, img_uvs_train = get_feature('hogwarts', x_train, y_train, img_lengths_train, img_uvs_train, 2, 6)
# x_train, y_train, img_lengths_train, img_uvs_train = get_feature('slope', x_train, y_train, img_lengths_train, img_uvs_train, 2, 12)
# x_train, y_train, img_lengths_train, img_uvs_train = get_feature('snow_grass', x_train, y_train, img_lengths_train, img_uvs_train, 2, 6)

x_train, y_train, img_lengths_train, img_uvs_train = get_feature('hogwarts', x_train, y_train, img_lengths_train, img_uvs_train)
# x_train, y_train, img_lengths_train, img_uvs_train = get_feature('slope', x_train, y_train, img_lengths_train, img_uvs_train)
# x_train, y_train, img_lengths_train, img_uvs_train = get_feature('snow_grass', x_train, y_train, img_lengths_train, img_uvs_train)

## testing set
img_lengths_test = []
img_uvs_test = []
img_lengths_test.append(0)
# x_test, y_test, img_lengths_test, img_uvs_test = get_feature('hogwarts', x_test, y_test, img_lengths_test, img_uvs_test, 6, 12)
# x_test, y_test, img_lengths_test, img_uvs_test = get_feature('slope', x_test, y_test, img_lengths_test, img_uvs_test, 6, 12)
# x_test, y_test, img_lengths_test, img_uvs_test = get_feature('snow_grass', x_test, y_test, img_lengths_test, img_uvs_test, 6, 12)

# x_test, y_test, img_lengths_test, img_uvs_test = get_feature('hogwarts', x_test, y_test, img_lengths_test, img_uvs_test)
x_test, y_test, img_lengths_test, img_uvs_test = get_feature('slope', x_test, y_test, img_lengths_test, img_uvs_test)
x_test, y_test, img_lengths_test, img_uvs_test = get_feature('snow_grass', x_test, y_test, img_lengths_test, img_uvs_test)


################################ normolize data set ############################

print len(x_train[0])
feature_norms = []
for i in range(len(x_train[0])):
    feature = np.zeros( [len(x_train), 1], dtype=np.float32 )
    for j in range(len(x_train)):
        feature[j] = x_train[j][i]

    norm = []
    norm.append(np.mean(feature))
    norm.append(np.std(feature))
    feature_norms.append(norm)
print feature_norms

x_train = normolize_data(x_train, feature_norms)
x_test = normolize_data(x_test, feature_norms)

for name, clf in zip(names, classifiers):
    print name
    # clf.fit(x_train, y_train)
    # # # scores = clf.score(x_test, y_test)
    # predict = clf.predict(x_test)
    # print metrics.confusion_matrix(y_test, predict)
    # print metrics.classification_report(y_test, predict)

    # classifier = OneVsRestClassifier(clf)
    classifier = OneVsRestClassifier(clf)
    classifier.fit(x_train, y_train)
    # y_prob = classifier.predict_proba(x_test)
    # y_labels = classifier.predict(x_test)
    # print y_prob, y_labels
    print x_test.ravel()

# y_vote = predict[:]
# for img_index in range(len(img_lengths_test)-1):
#     print img_index, len(img_lengths_test)-1
# #############################################################################
# ### draw result for one image
#     true_img = np.zeros([540,960], dtype=np.uint8)
#     predict_img = np.zeros([540,960], dtype=np.uint8)
#     new_predit_img = np.zeros([540,960], dtype=np.uint8)

#     x_one_img = x_test[img_lengths_test[img_index]:img_lengths_test[img_index+1]]
#     y_one_img = y_test[img_lengths_test[img_index]:img_lengths_test[img_index+1]]
#     uv_one_img = img_uvs_test[img_lengths_test[img_index]:img_lengths_test[img_index+1]]

#     predict_one_img = clf.predict(x_one_img)
#     # print 'before voting'
#     # print metrics.confusion_matrix(y_one_img, predict_one_img)
#     # print metrics.classification_report(y_one_img, predict_one_img)

#     for i in range(len(x_one_img)):
#         row = int(uv_one_img[i][-2])
#         col = int(uv_one_img[i][-1]) 

#         radius = 13.0/540.0 * row 

#         cv2.circle(true_img, (col, row), int(radius), y_one_img[i] * 50, -1)
#         cv2.circle(predict_img, (col, row), int(radius), predict_one_img[i] * 50, -1)
#         # true_img[row,col] = y_one_img[i] * 50
#         # predict_img[row,col] = predict_one_img[i] * 50
#         window_size = 30 + 100/540 * row
#         new_predit = voted_prediction(predict_img, row, col, 50, predict_one_img[i])
#         cv2.circle(new_predit_img, (col, row), int(radius), new_predit * 50, -1)
#         predict_one_img[i] = new_predit
    
#     y_vote[img_lengths_test[img_index]:img_lengths_test[img_index+1]] = predict_one_img


#     # cv2.imshow("true", true_img)
#     # cv2.imshow("predict_img", predict_img)
#     # cv2.imshow("voted_img", new_predit_img)
#     # cv2.waitKey(0)
# print 'after voting'
# print metrics.confusion_matrix(y_test, y_vote)
# print metrics.classification_report(y_test, y_vote)
# ###########################################################################

# # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


