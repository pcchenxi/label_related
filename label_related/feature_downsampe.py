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


names = [
        # "Nearest Neighbors",
        # "Linear SVM", "RBF SVM",
         "Decision Tree", 
         "Random Forest", 
        #  "Neural Net", 
        #  "AdaBoost",
        #  "Naive Bayes"
         ]

classifiers = [
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=15),   
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


def get_subfeature(img, type = 'c'):
    f = []
    label = img[:,:,0].sum()
    if label == 0:
        return f

    # num = cv2.countNonZero(img[:,:,0])
    # height_diff = img[:,:,2].sum()
    # slope = img[:,:,3].sum()
    # roughness = img[:,:,4].sum()
    # # vision_l = img[:,:,6].argmax()
    # vision_l = img[:,:,6].max()

    # if type == 'c':
    #     f.append(height_diff/num)
    #     f.append(slope/num)
    #     f.append(roughness/num)
    #     f.append(vision_l)
    # elif type == 'v':
    #     f.append(vision_l)
    # elif type == 'v':
    #     f.append(height_diff/num)
    #     f.append(slope/num)
    #     f.append(roughness/num)    

    height_diff = img[:,:,2].max()
    slope = img[:,:,3].max()
    roughness = img[:,:,4].max()
    vision_l = img[:,:,6].max()

    if type == 'c':
        f.append(height_diff)
        f.append(slope)
        f.append(roughness)
        f.append(vision_l)
    elif type == 'v':
        f.append(vision_l)
    elif type == 'g':
        f.append(height_diff)
        f.append(slope)
        f.append(roughness)
    return f


def get_features(img, row, col, radius, type = 'c'):
    x = []
    y = []

    ## use center
    if type == 'c':
        x.append(img[row, col, 2])
        x.append(img[row, col, 3])
        x.append(img[row, col, 4])
        x.append(img[row, col, 6])
    elif type == 'v':
        x.append(img[row, col, 6])
    elif type == 'g':
        x.append(img[row, col, 2])
        x.append(img[row, col, 3])
        x.append(img[row, col, 4])
    # y.append(img[row, col, 0])
    # return x, y


    start_row = row - radius/2
    end_row   = row + radius/2
    start_col = col - radius/2
    end_col   = col + radius/2

    img_rows, img_cols = img[:, :, 0].shape

    if start_row < 0:
        start_row = 0
    if start_col < 0:
        start_col = 0

    if end_row > img_rows:
        end_row = img_rows
    if end_col > img_cols:
        end_col = img_cols        

    mid_row = (start_row + end_row)/2
    mid_col = (start_col + end_col)/2
    
    sub_img_1 = img[start_row:mid_row,  start_col:mid_col]
    sub_img_2 = img[mid_row:end_row,    start_col:mid_col]
    sub_img_3 = img[start_row:mid_row,  mid_col:end_col]
    sub_img_4 = img[mid_row:end_row,    mid_col:end_col]

    # print sub_img_1[:,:,0].shape
    x.extend(get_subfeature(sub_img_1, type))
    x.extend(get_subfeature(sub_img_2, type))
    x.extend(get_subfeature(sub_img_3, type))
    x.extend(get_subfeature(sub_img_4, type))

    # x.append(img[row, col, 6])
    y.append(img[row, col, 0])

    if type == 'c':
        feature_size = 20
    elif type == 'v':
        feature_size = 5
    elif type == 'g':
        feature_size = 15       

    if len(x) != feature_size or y[0] > 3 or y[0] < 1:
        return [], []
    return x, y

def get_feature_mat(rosbag_name, img_index):
    x = []
    y = []

    features_mat = np.zeros([540,960, 7], dtype=np.float32)
    label_mat = np.zeros([540,960], dtype=np.uint8)
    label_mat_point = np.zeros([540,960], dtype=np.uint8)

    img_hd    = np.zeros([540,960], dtype=np.uint8)
    img_slope = np.zeros([540,960], dtype=np.uint8)
    img_rough = np.zeros([540,960], dtype=np.uint8)

    base_path_geo_result = '/home/xi/workspace/labels/output/' + rosbag_name + '/' + rosbag_name + '_'
    file_path = base_path_geo_result + str(img_index) + '_features.txt'
    print file_path
    with open(file_path) as f:
        content = f.readlines()
        for line in content:
            features = line.split(' ')
            features_clean = []

            for feature in features:
                feature = feature.replace('\n', '')
                features_clean.append(float(feature))

            if features_clean[0] < 1 or features_clean[0] > 3:
                continue

            
            # if features_clean[2] > 0.2 or features_clean[3] > 0.4:
            #     # print features_clean
            #     features_clean[0] = 3

            # if features_clean[0] == 1 or features_clean[0] == 2:
            #     features_clean[0] = 1

            row = int(features_clean[-2])
            col = int(features_clean[-1])

            features_mat[row, col, 0] = features_clean[0]     # true label
            features_mat[row, col, 1] = features_clean[1]     # height
            features_mat[row, col, 2] = features_clean[2]     # height difference
            features_mat[row, col, 3] = features_clean[3]     # slope
            features_mat[row, col, 4] = features_clean[4]     # roughness
            features_mat[row, col, 5] = features_clean[5]     # dist
            features_mat[row, col, 6] = features_clean[6]     # vision label

            label_mat_point[row, col] = features_clean[0] * 50

            d = features_mat[row, col, 5]
            # radius = 200*0.2/d
            radius = 20.0/540.0 * row
            cv2.circle(img_hd, (col, row), int(radius), features_clean[2] * 200 + 50, -1)
            cv2.circle(img_slope, (col, row), int(radius), features_clean[3] * 250+50, -1)
            cv2.circle(img_rough, (col, row), int(radius), features_clean[4] * 250+50, -1)


    # cv2.imshow("label_mat_point", label_mat_point)
    cv2.imshow("img_hd", img_hd)
    cv2.imshow("img_slope", img_slope)
    cv2.imshow("img_rough", img_rough)

    cv2.imwrite("img_hd.png", img_hd)
    cv2.imwrite("img_slope.png", img_slope)
    cv2.imwrite("img_rough.png", img_rough)
    cv2.waitKey(0)
##############################################################
    # label = features_mat[:,:,0]
    rows,cols = features_mat[:, :, 0].shape 
    resize_scale = 2
    resized_label = np.zeros([rows,cols], dtype=np.uint8)
    v_label = np.zeros([rows,cols], dtype=np.uint8)

    # print features_mat[:, :, 5].min(), features_mat[:, :, 5].max()

    for row in xrange(540):
        for col in xrange(960):
            if features_mat[row,col,0] == 0: 
                continue
            d = features_mat[row, col, 5]
            radius = 530*0.2/d
            # print d, radius
            x_sample, y_sample = get_features(features_mat, row, col, int(radius), 'c')

            # x_sample = []
            # y_sample = []
            # x_sample.append(features_mat[row, col, 2])
            # x_sample.append(features_mat[row, col, 3])
            # x_sample.append(features_mat[row, col, 4])
            # x_sample.append(features_mat[row, col, 6])
            # y_sample.append(features_mat[row, col, 0])

            # print x_sample


            if len(x_sample) == 0:
                continue

            # print x_sample, y_sample
            resized_label[row, col] = y_sample[0] * 50
            v_label[row, col] = features_mat[row,col,6]
            # print x_sample[-1], features_mat[row,col,6]
            # cv2.circle(label_mat_point, (col, row), int(radius), new_predit * 50, -1)

            x.append(x_sample)
            y.extend(y_sample)

    # print len(x)
            # print x_sample, y_sample

    # cv2.imshow("label_mat_point", label_mat_point)
    # cv2.imshow("resized_label", resized_label)
    # cv2.imshow("v_label", v_label)
    # cv2.waitKey(0)
    return x, y


################################################### get trainning and testing samples ####################################3
x_train = []
y_train = []
x_test = []
y_test = []

# for i in range(2, 12):
#     x_new, y_new = get_feature_mat('hogwarts', i)
#     x_train.extend(x_new)
#     y_train.extend(y_new)

for i in range(2, 12):
    x_new, y_new = get_feature_mat('trash_summer', i)
    x_train.extend(x_new)
    y_train.extend(y_new)

# print y_train

# for i in range(2, 12):
#     x_new, y_new = get_feature_mat('parking', i)
#     x_test.extend(x_new)
#     y_test.extend(y_new)

# for i in range(2, 12):
#     x_new, y_new = get_feature_mat('slope', i)
#     x_test.extend(x_new)
#     y_test.extend(y_new)

for i in range(2, 12):
    x_new, y_new = get_feature_mat('trash_summer', i)
    x_test.extend(x_new)
    y_test.extend(y_new)


# for i in range(2, 12):
#     x_new, y_new = get_feature_mat('trash_winter', i)
#     x_test.extend(x_new)
#     y_test.extend(y_new)

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




################################################### apply classifier ####################################3
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

y_test = label_binarize(y_test, classes=[1,2,3])
y_train= label_binarize(y_train, classes=[1,2,3])
n_classes = y_test.shape[1]

for name, clf in zip(names, classifiers):
    print name
    clf.fit(x_train, y_train)
    # # scores = clf.score(x_test, y_test)
    # predict = clf.predict(x_test)
    # confusion_mat = metrics.confusion_matrix(y_test, predict)

    # con_mat = np.zeros(confusion_mat.shape, dtype=np.float)
    # size = len(confusion_mat[:,0])
    # for i in xrange(size):
    #     row = confusion_mat[i,:]
    #     con_mat[i,:] = confusion_mat[i,:]/float(row.sum())
    # print con_mat
    # print metrics.classification_report(y_test, predict)


    classifier = OneVsRestClassifier(clf)
    classifier.fit(x_train, y_train)
    predict = classifier.predict(x_test)
    y_score = classifier.predict_proba(x_test)
    y_labels = classifier.predict(x_test)

    print metrics.classification_report(y_test, predict)
    # print metrics.confusion_matrix(y_test, y_labels)
    # print y_score, y_labels


    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



plt.plot(fpr["micro"], tpr["micro"],
        label='Geometric ROC curve (area = {0:0.2f})'
            ''.format(roc_auc["micro"]),
        color='darkorange', linestyle='-', linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()