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


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


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


def get_feature(rosbag_name, x, y, img_lengths, img_uvs):
    base_path_geo_result = '/home/xi/workspace/labels/output/' + rosbag_name + '/' + rosbag_name + '_'
    for i in range(2,12):   
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

                x_new = features_clean[1:6]
                y_new = int(features_clean[0])

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
img_uvs = []

img_lengths = []
img_lengths.append(0)
x, y, img_lengths, img_uvs = get_feature('snow_grass', x, y, img_lengths, img_uvs)
x, y, img_lengths, img_uvs = get_feature('hogwarts', x, y, img_lengths, img_uvs)
x, y, img_lengths, img_uvs = get_feature('slope', x, y, img_lengths, img_uvs)

# print img_lengths, len(x)

x = preprocessing.scale(x)
print 'finish preprocessing'

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50, random_state=40)

# ########################################
# # manuallly split 
# for i in range(1, len(x)):
#     if y[i] > 3 or y[i] < 1:
#         continue 

#     if i%2 == 0:
#         x_train.append(x[i])
#         y_train.append(y[i])
#     else:
#         x_test.append(x[i])
#         y_test.append(y[i])
# ##

############################################################################################


print len(y)

y_vote = y[:]
for name, clf in zip(names, classifiers):
    print name
    clf.fit(x_train, y_train)
    # # scores = clf.score(x_test, y_test)
    # all_predict = clf.predict(x)
    # print metrics.confusion_matrix(y, all_predict)
    # print metrics.classification_report(y, all_predict)

    all_predict = clf.predict(x_test)
    print metrics.confusion_matrix(y_test, all_predict)
    print metrics.classification_report(y_test, all_predict)
    joblib.dump(clf, name + '_model.pkl') 
    # # print len(x), len(x_train), len(x_test)
    # # scores = cross_val_score(clf, x, y, cv=5)
    # # print name, scores


    # for img_index in range(len(img_lengths)-1):
    #     print img_index, len(img_lengths)-1
    # #############################################################################
    # ### draw result for one image
    #     true_img = np.zeros([540,960], dtype=np.uint8)
    #     predict_img = np.zeros([540,960], dtype=np.uint8)
    #     new_predit_img = np.zeros([540,960], dtype=np.uint8)
    #     # img_index = 4

    #     x_one_img = x[img_lengths[img_index]:img_lengths[img_index+1]]
    #     y_one_img = y[img_lengths[img_index]:img_lengths[img_index+1]]
    #     uv_one_img = img_uvs[img_lengths[img_index]:img_lengths[img_index+1]]

    #     # print y

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
    #         window_size = 30 + 200/540 * row
    #         new_predit = voted_prediction(predict_img, row, col, window_size, predict_one_img[i])
    #         cv2.circle(new_predit_img, (col, row), int(radius), new_predit * 50, -1)
    #         predict_one_img[i] = new_predit
        
    #     y_vote[img_lengths[img_index]:img_lengths[img_index+1]] = predict_one_img
    # print 'after voting'
    # print metrics.confusion_matrix(y, y_vote)
    # print metrics.classification_report(y, y_vote)
    #############################################################################
    break
    # # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

cv2.imshow("true", true_img)
cv2.imshow("predict_img", predict_img)
cv2.imshow("voted_img", new_predit_img)
cv2.waitKey(0)


#######################################################################

# # # clf = svm.SVC(kernel='rbf', C=1, gamma='auto')
# # clf = svm.SVC(kernel='linear', cache_size=1000)
# clf = DecisionTreeClassifier()
# # clf = KNeighborsClassifier(n_neighbors=100)

# # clf.fit(x_train,y_train)
# # joblib.dump(clf, 'svm_hogwarts_123_model.pkl') 
# # print "dome"


# scores = cross_val_score(clf, x, y, cv=5)
# print scores
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
