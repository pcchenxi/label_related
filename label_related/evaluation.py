import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import cv2
import time
import numpy as np
import pickle
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

plt.rc('text', usetex=True)
plt.rc('font', family='serif')



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
        #  "Decision Tree", 
         "Random Forest", 
        #  "Neural Net", 
        #  "AdaBoost",
        #  "Naive Bayes"
         ]

classifiers = [
    # KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    # DecisionTreeClassifier(max_depth=4),
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

def normolize_dataset(x):
    feature_norms = []
    for i in range(len(x[0])):
        feature = np.zeros( [len(x), 1], dtype=np.float32 )
        for j in range(len(x)):
            feature[j] = x[j][i]

        norm = []
        norm.append(np.mean(feature))
        norm.append(np.std(feature))
        feature_norms.append(norm)
    
    return feature_norms


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


def get_feature(rosbag_name, x, y, img_lengths, img_uvs, type = 'c', index_start = 2, index_end = 12):
    base_path_geo_result = '/home/xi/workspace/labels/output/' + rosbag_name + '/' + rosbag_name + '_'
    pickle_path = '/home/xi/workspace/labels/' + rosbag_name + '/' + rosbag_name + '_'
    total_sample = 0
    safe_sample = 0
    risky_sample = 0
    obstacle_sample = 0

    for i in range(index_start, index_end):   
        pick = pickle.load( open( pickle_path + str(i) + '.p', "rb" ) )
        print pick.shape
        file_path = base_path_geo_result + str(i) + '_features.txt'
        print file_path
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
                r   = features_clean[4]
                d   = features_clean[5]

                row = int(features_clean[7]*255/540)
                col = int(features_clean[8]*512/960)

            
                # if features_clean[2] > 0.3:
                #     features_clean[0] = 3

                x_new = []

                feature_v = []
                for v_index in xrange(34):
                    f_v = pick[-1][row][col][v_index]
                    # print f_v
                    feature_v.append(f_v)

                if type == 'g':
                    x_new = features_clean[2:5]
                elif type == 'v':
                    # x_new.append(features_clean[6]) # vision
                    x_new.extend(feature_v)
                elif type == 'c':
                    x_new = features_clean[2:5]
                    x_new.extend(feature_v)
                    # x_new.append(features_clean[6]) # vision

                y_new = int(features_clean[0]) # true label

                x.append(x_new)
                y.append(y_new)

                uv = features_clean
                img_uvs.append(uv)

                total_sample += 1
                if y_new == 1:
                    safe_sample += 1
                elif y_new == 2:
                    risky_sample += 1
                elif y_new == 3:
                    obstacle_sample += 1
                # img_uvs.append(i)


    # print total_sample, safe_sample, risky_sample, obstacle_sample
    return x, y, img_lengths, img_uvs

 
def get_result(x_train, x_test, y_train, y_test):
    y_test = label_binarize(y_test, classes=[1,2,3])
    y_train= label_binarize(y_train, classes=[1,2,3])
    n_classes = y_test.shape[1]

    for name, clf in zip(names, classifiers):

        print name
        # clf.fit(x_train, y_train)
        # # scores = clf.score(x_test, y_test)
        # predict = clf.predict(x_test)
        # print metrics.confusion_matrix(y_test, predict)
        # print metrics.classification_report(y_test, predict)

        classifier = OneVsRestClassifier(clf)
        classifier.fit(x_train, y_train)
        predict = classifier.predict(x_test)
        y_score = classifier.predict_proba(x_test)
        y_labels = classifier.predict(x_test)

        print metrics.classification_report(y_test, predict)
        # # print metrics.confusion_matrix(y_test, y_labels)
        # # print y_score, y_labels


        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area    
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc

def draw_single_ROC(fpr_c, tpr_c, roc_auc_c, fpr_v, tpr_v, roc_auc_v, fpr_g, tpr_g, roc_auc_g, type='risky'):
    plt.figure()
    plt.plot(fpr_c[0], tpr_c[0],
            label='Fusion (area = {0:0.2f})'
                ''.format(roc_auc_c[0]),
            color='deeppink', linestyle='-', linewidth=4)            
    plt.plot(fpr_g[0], tpr_g[0],
            label='Geometry (area = {0:0.2f})'
                ''.format(roc_auc_g[0]),
            color='cornflowerblue', linestyle='-', linewidth=4)
    plt.plot(fpr_v[0], tpr_v[0],
            label='Vision (area = {0:0.2f})'
                ''.format(roc_auc_v[0]),
            color='darkorange', linestyle='-', linewidth=4)    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Classification result on ' + type + ' terrain')
    plt.legend(loc="lower right")
    plt.show()

def draw_ROC(fpr_c, tpr_c, roc_auc_c, fpr_v, tpr_v, roc_auc_v, fpr_g, tpr_g, roc_auc_g):
    # flat terrain
    plt.figure()
    plt.plot(fpr_c[0], tpr_c[0],
            label='Fusion (area = {0:0.2f})'
                ''.format(roc_auc_c[0]),
            color='deeppink', linestyle='-', linewidth=4)            
    plt.plot(fpr_g[0], tpr_g[0],
            label='Geometry (area = {0:0.2f})'
                ''.format(roc_auc_g[0]),
            color='cornflowerblue', linestyle='-', linewidth=4)
    plt.plot(fpr_v[0], tpr_v[0],
            label='Vision (area = {0:0.2f})'
                ''.format(roc_auc_v[0]),
            color='darkorange', linestyle='-', linewidth=4)    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Classification result on safe terrain ')
    plt.legend(loc="lower right")
    plt.show()

    # risky terrain
    plt.figure()
    plt.plot(fpr_c[1], tpr_c[1],
            label='Fusion (area = {0:0.2f})'
                ''.format(roc_auc_c[1]),
            color='deeppink', linestyle='-', linewidth=4)            
    plt.plot(fpr_g[1], tpr_g[1],
            label='Geometry (area = {0:0.2f})'
                ''.format(roc_auc_g[1]),
            color='cornflowerblue', linestyle='-', linewidth=4)
    plt.plot(fpr_v[1], tpr_v[1],
            label='Vision (area = {0:0.2f})'
                ''.format(roc_auc_v[1]),
            color='darkorange', linestyle='-', linewidth=4)   
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Classification result on risky terrain ')
    plt.legend(loc="lower right")
    plt.show()

    # obstacle terrain
    plt.figure()
    plt.plot(fpr_c[2], tpr_c[2],
            label='Fusion (area = {0:0.2f})'
                ''.format(roc_auc_c[2]),
            color='deeppink', linestyle='-', linewidth=4)            
    plt.plot(fpr_g[2], tpr_g[2],
            label='Geometry (area = {0:0.2f})'
                ''.format(roc_auc_g[2]),
            color='cornflowerblue', linestyle='-', linewidth=4)
    plt.plot(fpr_v[2], tpr_v[2],
            label='Vision (area = {0:0.2f})'
                ''.format(roc_auc_v[2]),
            color='darkorange', linestyle='-', linewidth=4)   
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Classification result on obstacle ')
    plt.legend(loc="lower right")
    plt.show()


def draw_result_img(x_test, y_test, x_train, y_train, img_lengths_test, img_uvs_test, type):
    # y_vote = predict[:]
    clf = RandomForestClassifier(max_depth=5, n_estimators=15)
    # classifier = OneVsRestClassifier(clf)
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)

    for img_index in range(len(img_lengths_test)-1):
        print img_index, len(img_lengths_test)-1
        #############################################################################
        ### draw result for one image
        true_img = np.zeros([540,960, 3], dtype=np.uint8)
        predict_img = np.zeros([540,960, 3], dtype=np.uint8)
        new_predit_img = np.zeros([540,960], dtype=np.uint8)

        x_one_img = x_test[img_lengths_test[img_index]:img_lengths_test[img_index+1]]
        y_one_img = y_test[img_lengths_test[img_index]:img_lengths_test[img_index+1]]
        uv_one_img = img_uvs_test[img_lengths_test[img_index]:img_lengths_test[img_index+1]]

        predict_one_img = clf.predict(x_one_img)
        # print 'before voting'
        # print metrics.confusion_matrix(y_one_img, predict_one_img)
        # print metrics.classification_report(y_one_img, predict_one_img)

        for i in range(len(x_one_img)):
            row = int(uv_one_img[i][-2])
            col = int(uv_one_img[i][-1]) 

            radius = 20.0/540.0 * row 

            # cv2.circle(true_img, (col, row), int(radius), y_one_img[i] * 50, -1)
            if y_one_img[i] == 0:
                cv2.circle(true_img, (col, row), int(radius), [0, 0, 200], -1)
            if y_one_img[i] == 1:
                cv2.circle(true_img, (col, row), int(radius), [0, 200, 0], -1)
            if y_one_img[i] == 2:
                cv2.circle(true_img, (col, row), int(radius), [200, 0, 0], -1)

            # cv2.circle(predict_img, (col, row), int(radius), predict_one_img[i] * 50, -1)
            # true_img[row,col] = y_one_img[i] * 50
            # predict_img[row,col] = predict_one_img[i] * 50
            # window_size = 30 + 100/540 * row
            # new_predit = voted_prediction(predict_img, row, col, 50, predict_one_img[i])
            # cv2.circle(predict_img, (col, row), int(radius), predict_one_img[i] * 50, -1)
            if predict_one_img[i] == 1:
                cv2.circle(predict_img, (col, row), int(radius), [0, 255, 0], -1)
            if predict_one_img[i] == 2:
                cv2.circle(predict_img, (col, row), int(radius), [0, 255, 255], -1)
            if predict_one_img[i] == 3:
                cv2.circle(predict_img, (col, row), int(radius), [0, 0, 255], -1)

            # predict_one_img[i] = new_predit
        
        # y_vote[img_lengths_test[img_index]:img_lengths_test[img_index+1]] = predict_one_img

        path = type + str(img_index) + '.png'
        path_t = type + str(img_index) + '_t.png'
        # cv2.imshow("true", true_img)
        # cv2.imshow("predict_img", predict_img)
        cv2.imwrite(path, predict_img)
        # cv2.imwrite(path_t, true_img)
        # cv2.imshow("voted_img", new_predit_img)
        # cv2.waitKey(0)
    ###########################################################################


x = []
y = []
x_train_c = []
y_train_c = []
x_test_c = []
y_test_c = []

x_train_v = []
y_train_v = []
x_test_v = []
y_test_v = []

x_train_g = []
y_train_g = []
x_test_g = []
y_test_g = []
################################ get trainning and test set ############################

img_lengths_train_c = []
img_uvs_train_c = []
img_lengths_train_g = []
img_uvs_train_g = []
img_lengths_train_v = []
img_uvs_train_v = []

img_lengths_train_c.append(0)
img_lengths_train_g.append(0)
img_lengths_train_v.append(0)

# x_train, y_train, img_lengths_train, img_uvs_train = get_feature('hogwarts', x_train, y_train, img_lengths_train, img_uvs_train)

data_1 = 'hogwarts'
data_2 = 'snow_grass'
data_3 = 'slope'
data_4 = 'parking'
data_5 = 'trash_summer'
data_6 = 'trash_winter'
data_7 = 'stain'

s_1 = 2
c_1 = 12
e_1 = 12

s_2 = 2
c_2 = 2
e_2 = 12

# x_train_c, y_train_c, img_lengths_train_c, img_uvs_train_c = get_feature(data_5, x_train_c, y_train_c, img_lengths_train_c, img_uvs_train_c, 'c', s_1, c_1)
# x_train_v, y_train_v, img_lengths_train_v, img_uvs_train_v = get_feature(data_5, x_train_v, y_train_v, img_lengths_train_v, img_uvs_train_v, 'v', s_1, c_1)
# x_train_g, y_train_g, img_lengths_train_g, img_uvs_train_g = get_feature(data_5, x_train_g, y_train_g, img_lengths_train_g, img_uvs_train_g, 'g', s_1, c_1)

x_train_c, y_train_c, img_lengths_train_c, img_uvs_train_c = get_feature(data_1, x_train_c, y_train_c, img_lengths_train_c, img_uvs_train_c, 'c', s_1, c_1)
x_train_v, y_train_v, img_lengths_train_v, img_uvs_train_v = get_feature(data_1, x_train_v, y_train_v, img_lengths_train_v, img_uvs_train_v, 'v', s_1, c_1)
x_train_g, y_train_g, img_lengths_train_g, img_uvs_train_g = get_feature(data_1, x_train_g, y_train_g, img_lengths_train_g, img_uvs_train_g, 'g', s_1, c_1)

x_train_c, y_train_c, img_lengths_train_c, img_uvs_train_c = get_feature(data_2, x_train_c, y_train_c, img_lengths_train_c, img_uvs_train_c, 'c', s_1, c_1)
x_train_v, y_train_v, img_lengths_train_v, img_uvs_train_v = get_feature(data_2, x_train_v, y_train_v, img_lengths_train_v, img_uvs_train_v, 'v', s_1, c_1)
x_train_g, y_train_g, img_lengths_train_g, img_uvs_train_g = get_feature(data_2, x_train_g, y_train_g, img_lengths_train_g, img_uvs_train_g, 'g', s_1, c_1)

# x_train_c, y_train_c, img_lengths_train_c, img_uvs_train_c = get_feature(data_4, x_train_c, y_train_c, img_lengths_train_c, img_uvs_train_c, 'c', 2, 8)
# x_train_v, y_train_v, img_lengths_train_v, img_uvs_train_v = get_feature(data_4, x_train_v, y_train_v, img_lengths_train_v, img_uvs_train_v, 'v', 2, 8)
# x_train_g, y_train_g, img_lengths_train_g, img_uvs_train_g = get_feature(data_4, x_train_g, y_train_g, img_lengths_train_g, img_uvs_train_g, 'g', 2, 8)

# x_train_c, y_train_c, img_lengths_train, img_uvs_train = get_feature(data_6, x_train_c, y_train_c, img_lengths_train, img_uvs_train, 'c', s_2, c_2)
# x_train_v, y_train_v, img_lengths_train, img_uvs_train = get_feature(data_6, x_train_v, y_train_v, img_lengths_train, img_uvs_train, 'v', s_2, c_2)
# x_train_g, y_train_g, img_lengths_train, img_uvs_train = get_feature(data_6, x_train_g, y_train_g, img_lengths_train, img_uvs_train, 'g', s_2, c_2)

# x_train_c, y_train_c, img_lengths_train, img_uvs_train = get_feature(data_5, x_train_c, y_train_c, img_lengths_train, img_uvs_train, 'c', s_2, c_2)
# x_train_v, y_train_v, img_lengths_train, img_uvs_train = get_feature(data_5, x_train_v, y_train_v, img_lengths_train, img_uvs_train, 'v', s_2, c_2)
# x_train_g, y_train_g, img_lengths_train, img_uvs_train = get_feature(data_5, x_train_g, y_train_g, img_lengths_train, img_uvs_train, 'g', s_2, c_2)


## testing set
img_lengths_test_c = []
img_uvs_test_c = []
img_lengths_test_g = []
img_uvs_test_g = []
img_lengths_test_v = []
img_uvs_test_v = []
img_lengths_test_c.append(0)
img_lengths_test_g.append(0)
img_lengths_test_v.append(0)

# x_test, y_test, img_lengths_test, img_uvs_test = get_feature('slope', x_test, y_test, img_lengths_test, img_uvs_test)
# x_test, y_test, img_lengths_test, img_uvs_test = get_feature('snow_grass', x_test, y_test, img_lengths_test, img_uvs_test)

# x_test_c, y_test_c, img_lengths_test, img_uvs_test = get_feature(data_1, x_test_c, y_test_c, img_lengths_test, img_uvs_test, 'c', c_1, e_1)
# x_test_v, y_test_v, img_lengths_test, img_uvs_test = get_feature(data_1, x_test_v, y_test_v, img_lengths_test, img_uvs_test, 'v', c_1, e_1)
# x_test_g, y_test_g, img_lengths_test, img_uvs_test = get_feature(data_1, x_test_g, y_test_g, img_lengths_test, img_uvs_test, 'g', c_1, e_1)

# x_test_c, y_test_c, img_lengths_test, img_uvs_test = get_feature(data_2, x_test_c, y_test_c, img_lengths_test, img_uvs_test, 'c', c_2, e_3)
# x_test_v, y_test_v, img_lengths_test, img_uvs_test = get_feature(data_2, x_test_v, y_test_v, img_lengths_test, img_uvs_test, 'v', c_2, e_3)
# x_test_g, y_test_g, img_lengths_test, img_uvs_test = get_feature(data_2, x_test_g, y_test_g, img_lengths_test, img_uvs_test, 'g', c_2, e_3)


x_test_c, y_test_c, img_lengths_test_c, img_uvs_test_c = get_feature(data_3, x_test_c, y_test_c, img_lengths_test_c, img_uvs_test_c, 'c')
x_test_v, y_test_v, img_lengths_test_v, img_uvs_test_v = get_feature(data_3, x_test_v, y_test_v, img_lengths_test_v, img_uvs_test_v, 'v')
x_test_g, y_test_g, img_lengths_test_g, img_uvs_test_g = get_feature(data_3, x_test_g, y_test_g, img_lengths_test_g, img_uvs_test_g, 'g')

# x_test_c, y_test_c, img_lengths_test_c, img_uvs_test_c = get_feature(data_4, x_test_c, y_test_c, img_lengths_test_c, img_uvs_test_c, 'c')
# x_test_v, y_test_v, img_lengths_test_v, img_uvs_test_v = get_feature(data_4, x_test_v, y_test_v, img_lengths_test_v, img_uvs_test_v, 'v')
# x_test_g, y_test_g, img_lengths_test_g, img_uvs_test_g = get_feature(data_4, x_test_g, y_test_g, img_lengths_test_g, img_uvs_test_g, 'g')

# x_test_c, y_test_c, img_lengths_test_c, img_uvs_test_c = get_feature(data_4, x_test_c, y_test_c, img_lengths_test_c, img_uvs_test_c, 'c')
# x_test_v, y_test_v, img_lengths_test_v, img_uvs_test_v = get_feature(data_4, x_test_v, y_test_v, img_lengths_test_v, img_uvs_test_v, 'v')
# x_test_g, y_test_g, img_lengths_test_g, img_uvs_test_g = get_feature(data_4, x_test_g, y_test_g, img_lengths_test_g, img_uvs_test_g, 'g')

x_test_c, y_test_c, img_lengths_test_c, img_uvs_test_c = get_feature(data_5, x_test_c, y_test_c, img_lengths_test_c, img_uvs_test_c, 'c', c_2, e_2)
x_test_v, y_test_v, img_lengths_test_v, img_uvs_test_v = get_feature(data_5, x_test_v, y_test_v, img_lengths_test_v, img_uvs_test_v, 'v', c_2, e_2)
x_test_g, y_test_g, img_lengths_test_g, img_uvs_test_g = get_feature(data_5, x_test_g, y_test_g, img_lengths_test_g, img_uvs_test_g, 'g', c_2, e_2)

# x_test_c, y_test_c, img_lengths_test_c, img_uvs_test_c = get_feature(data_6, x_test_c, y_test_c, img_lengths_test_c, img_uvs_test_c, 'c', c_2, e_2)
# x_test_v, y_test_v, img_lengths_test_v, img_uvs_test_v = get_feature(data_6, x_test_v, y_test_v, img_lengths_test_v, img_uvs_test_v, 'v', c_2, e_2)
# x_test_g, y_test_g, img_lengths_test_g, img_uvs_test_g = get_feature(data_6, x_test_g, y_test_g, img_lengths_test_g, img_uvs_test_g, 'g', c_2, e_2)

################################ normolize data set ############################

feature_norms_c = normolize_dataset(x_train_c)
x_train_c       = normolize_data(x_train_c, feature_norms_c)
x_test_c        = normolize_data(x_test_c, feature_norms_c)

feature_norms_v = normolize_dataset(x_train_v)
x_train_v       = normolize_data(x_train_v, feature_norms_v)
x_test_v        = normolize_data(x_test_v, feature_norms_v)


feature_norms_g = normolize_dataset(x_train_g)
x_train_g       = normolize_data(x_train_g, feature_norms_g)
x_test_g        = normolize_data(x_test_g, feature_norms_g)


from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

fpr_c, tpr_c, roc_auc_c = get_result(x_train_c, x_test_c, y_train_c, y_test_c)
fpr_v, tpr_v, roc_auc_v = get_result(x_train_v, x_test_v, y_train_v, y_test_v)
fpr_g, tpr_g, roc_auc_g = get_result(x_train_g, x_test_g, y_train_g, y_test_g)

draw_ROC(fpr_c, tpr_c, roc_auc_c, fpr_v, tpr_v, roc_auc_v, fpr_g, tpr_g, roc_auc_g)

# fpr, tpr, roc_auc = get_result(x_train_c, x_test_c, y_train_c, y_test_c)
# plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],
#         label='Combined ROC curve (area = {0:0.2f})'
#             ''.format(roc_auc["micro"]),
#         color='deeppink', linestyle='-', linewidth=4)

x_train = x_train_v
y_train = y_train_v
x_test = x_test_v
y_test = y_test_v

for name, clf in zip(names, classifiers):
    print name
    # clf.fit(x_train, y_train)
    # # scores = clf.score(x_test, y_test)
    # predict = clf.predict(x_test)
    # print metrics.confusion_matrix(y_test, predict)
    # print metrics.classification_report(y_test, predict)

    # draw_result_img(x_test_c, y_test_c, x_train_c, y_train_c, img_lengths_test_c, img_uvs_test_c, 'c')
    # draw_result_img(x_test_v, y_test_v, x_train_v, y_train_v, img_lengths_test_v, img_uvs_test_v, 'v')
    # draw_result_img(x_test_g, y_test_g, x_train_g, y_train_g, img_lengths_test_g, img_uvs_test_g, 'g')

