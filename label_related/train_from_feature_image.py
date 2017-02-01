import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import cv2
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

## classifiers
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


names = ["Nearest Neighbors", 
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    #SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]



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

                if features_clean[0] < 1 or features_clean[0] > 3:
                    continue
                x_new = features_clean[1:5]
                y_new = int(features_clean[0])
                x.append(x_new)
                y.append(y_new)

    return x, y


x = []
y = []

x, y = get_feature('snow_grass', x, y)
# x, y = get_feature('hogwarts', x, y)

x = preprocessing.scale(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50, random_state=30)

for name, clf in zip(names, classifiers):
    clf.fit(x_train, y_train)
    scores = clf.score(x_test, y_test)
    # print len(x), len(x_train), len(x_test)
    # scores = cross_val_score(clf, x, y, cv=5)
    print name, scores
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))





# # clf = svm.SVC(kernel='rbf', C=1, gamma='auto')
# clf = svm.SVC(kernel='linear', cache_size=1000)
# clf = tree.DecisionTreeClassifier()
# clf = KNeighborsClassifier(n_neighbors=100)

# clf.fit(x,y)
# joblib.dump(clf, 'svm_hogwarts_123_model.pkl') 
# print "dome"


# scores = cross_val_score(clf, x, y, cv=5)
# print scores
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
