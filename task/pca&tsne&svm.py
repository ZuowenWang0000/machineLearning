from matplotlib.ticker import NullFormatter
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.decomposition import PCA
from sklearn import neural_network
from sklearn.metrics import accuracy_score
import sklearn
from sklearn.semi_supervised import LabelSpreading


train_labeled = pd.read_hdf("train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")

train_labeled = np.array(train_labeled)

features = train_labeled [:, 1:129]

lables = train_labeled[:,0]




X = features
y = lables
X2 = train_unlabeled

##SCALLING DATA##
print("feature scaled")
X = sklearn.preprocessing.scale(X)
X2 =sklearn.preprocessing.scale(X2)

###############################################
######MIN MAX SCALLING###################
# X = sklearn.preprocessing.MinMaxScaler().fit_transform(X)
# X2 = sklearn.preprocessing.MinMaxScaler().fit_transform(X2)

##pca the none-scaled-features to 50 dimensions
pca = PCA(n_components=3)
print("pca to 3 dimensions")
X = pca.fit_transform(X)
tempX = X
##pca the none-scaled unlabeled features to 50 dimensions
pca2 = PCA(n_components=3)
X2 = pca2.fit_transform(X2)

##tsne the none-scaled-features to 3 dimensions
# tsne = manifold.TSNE(n_components=3, init='random',verbose = 3,
#                          random_state=0, perplexity=20)
# print('tsne preplexity = 20')
# X = tsne.fit_transform(X)

##tsne the none-sclaed-unlabeled-features to 3 dimensions
# X2 = tsne.fit_transform(X2)

print('shape of labeled reduced dimension features: {Xshape}'.format(Xshape = X.shape))
print('shape of unlabeled reduced dimension features: {X2shape}'.format(X2shape = X2.shape))

#############SEMI SUPERVISED##############################################
# lp = LabelSpreading(kernel='knn', gamma=20, n_neighbors=7, alpha=0.1, max_iter=30, tol=0.0001, n_jobs=1)
#
# z = np.array([-1])
# print("here2")
#
# for i in range(21000):
#     y = np.concatenate((y, z), axis=0)
#
# X = np.concatenate((X,X2),axis=0)
#
# lp.fit(X,y)
# result = lp.predict(X)
#
# #@@@@@@@@@@@@@@@@@VALIDATE USING NN on 9000 data@@@@@@@@@@@@@@@@@@@@@@@@
# nn = neural_network.MLPClassifier(hidden_layer_sizes=(512, ),activation= 'relu', solver='adam', alpha=1,
# learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, shuffle=True,
#     random_state=None, tol=0.0001, verbose=True, max_iter = 400, early_stopping=False,momentum=0.9,
# nesterovs_momentum=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#
# nn.fit(X,result)
# vali = nn.predict(tempX)
# print(accuracy_score(lables,vali))

#######################################################

# X = sklearn.preprocessing.scale(X)
# X2 = sklearn.preprocessing.scale(X2)

# svc = sklearn.svm.SVC(C=10, kernel='rbf', degree=4, probability=False, tol=0.0001, cache_size=4000, class_weight=None, verbose=True, max_iter=-1, decision_function_shape="ovr", random_state=None)
# svc.fit(X,y)
# labelsOfUnlabeled = svc.predict(X2)
# print("lables of unlabeled {}".format(labelsOfUnlabeled.shape))
#
# np.savetxt('labelsOfUnlabeled.csv', labelsOfUnlabeled ,delimiter = ",")
# #
# print("rbf kernal , C = 10, ovr")
###########################################
# nn = neural_network.MLPClassifier(hidden_layer_sizes=(1024, ),activation= 'relu', solver='adam', alpha=1,
# learning_rate='adaptive', learning_rate_init=0.0001, power_t=0.5, shuffle=True,
#     random_state=None, tol=0.00001, verbose=True, max_iter = 400, early_stopping=False,momentum=0.9,
# nesterovs_momentum=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#
#
# nn.fit(X,y)
# labelsOfUnlabeled = nn.predict(X2)
# validate = nn.predict(sklearn.preprocessing.scale(features))
# trainingSetLabelPredict = nn.predict(X)

##############################################################


validate_Label = np.loadtxt(open('21000ValidationLabels.csv'))
#
# ######COMPARE THE RESULT######
#
# print(accuracy_score(validate_Label,labelsOfUnlabeled))

# print(accuracy_score(lables,trainingSetLabelPredict))

