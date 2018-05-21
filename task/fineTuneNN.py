import pandas as pd
import numpy as np
import sklearn
from sklearn import neural_network
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelPropagation
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelSpreading
from sklearn import datasets
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV

train_labeled = pd.read_hdf("train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")
benchmark8000 = np.loadtxt('HugoBenchMark.csv', delimiter = ',')

train_labeled = np.array(train_labeled)

features = train_labeled [:, 1:129]

lables = train_labeled[:,0]
trainingSetLabels = lables

X = sklearn.preprocessing.scale(features)
y = lables


clf1 = neural_network.MLPClassifier(hidden_layer_sizes=(1024,),activation= 'relu', solver='adam', alpha=0.1,
learning_rate='constant', learning_rate_init=0.001, power_t=0.5, shuffle=True,
    random_state=None, tol=0.1, verbose=False, momentum=0.9,
nesterovs_momentum=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# clf1 = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.00001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovo', random_state=2)

clf2 = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.00001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovo', random_state=2)


# clf3 = LabelSpreading(kernel="knn", gamma=20, n_neighbors=20, alpha=0.2, max_iter=300, tol=0.00001, n_jobs=-1)

clf3 = RandomForestClassifier(n_estimators=800, criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight=None)

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

###################################CROSS VALIDATION AND GRID SEARCH#####################################################
# for clf, name in zip([clf1, clf2, clf3, eclf], ['neural networks', 'svc', 'random forest', 'Ensemble']):
#     scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
#     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), name))


# params = {'lr__hidden_layer_sizes': [(256,),(512,)], 'rf__C':[1,5]}

# params = {'lr__hidden_layer_sizes': [(7, 7), (128,), (128, 7),(512,),(1024,)],'lr__tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],'lr__epsilon':[1e-3, 1e-7, 1e-8, 1e-9, 1e-8]}
#
# grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
# scores = cross_val_score(grid, X, y, cv=10, scoring='accuracy')
# print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std()))

# grid = grid.fit(X, y)
##########################################################################################################################
eclf.fit(X,y)
tempResult21000 = eclf.predict(sklearn.preprocessing.scale(train_unlabeled))

np.savetxt('tempRestul21000.csv',tempResult21000,delimiter = ',')

print("finished")
print(clf1)
print(clf2)
print(clf3)