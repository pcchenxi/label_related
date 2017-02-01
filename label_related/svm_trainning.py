import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import tree
from sklearn.externals import joblib

x = []
y = []

training_files = ["slope_1.txt", "slope_2.txt", "slope_3.txt"]

for file in training_files:
    i = 0    
    with open(file) as f:
        content = f.readlines()
        for line in content:
            # i=i+1
            # if i%1000 == 1:
            #     print i, len(content)
            # i=i+1
            # if i < len(content)*5/10:
            #     continue
            features = line.split(' ')
            
            features_clean = []
            for feature in features:
                feature = feature.replace('\n', '')
                features_clean.append(float(feature))
            # print features_clean

            x_new = features_clean[1:]
            y_new = int(features_clean[0]/50)
            # y_new = np.array(y_new).reshape((1, -1))
            x.append(x_new)
            y.append(y_new)


# clf = svm.SVC(kernel='rbf', C=1, gamma='auto')
clf = tree.DecisionTreeClassifier()

clf.fit(x,y)


joblib.dump(clf, 'tree_slope_12_model.pkl') 
print "dome"
