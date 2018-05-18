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

nn = neural_network.MLPClassifier(hidden_layer_sizes=(256, ),activation= 'relu', solver='adam', alpha=1,
learning_rate='adaptive', learning_rate_init=0.01, power_t=0.5, shuffle=True,
    random_state=None, tol=0.001, verbose=True, momentum=0.9,
nesterovs_momentum=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

nn.fit(X,y)
labelsOfUnlabeled = nn.predict(X2)

n = 10
m = 128
a = [0] * n
for i in range(n):
    a[i] = [0] * m
print(len(a))

for x in range (0,8999):
    if train_labeled[x,0]==0:
        a[0] = a[0]+ features[x,]
    elif train_labeled[x,0]==1:
        a[1] = a[1]+ features[x,]
    elif train_labeled[x,0]==2:
        a[2] = a[2]+ features[x,]
    elif train_labeled[x,0]==3:
        a[3] = a[3]+ features[x,]
    elif train_labeled[x,0]==4:
        a[4] = a[4]+ features[x,]
    elif train_labeled[x,0]==5:
        a[5] = a[5]+ features[x,]
    elif train_labeled[x,0]==6:
        a[6] = a[6]+ features[x,]
    elif train_labeled[x,0]==7:
        a[7] = a[7]+ features[x,]
    elif train_labeled[x,0]==8:
        a[8] = a[8]+ features[x,]
    elif train_labeled[x,0]==9:
        a[9] = a[9]+ features[x,]

a[0] = a[0]/866
a[1] = a[1]/983
a[2] = a[2]/903
a[3] = a[3]/925
a[4] = a[4]/818
a[5] = a[5]/793
a[6] = a[6]/917
a[7] = a[7]/954
a[8] = a[8]/928
a[9] = a[9]/913


classZero = np.empty([1,128])
classOne = np.empty([1,128])
classTwo = np.empty([1,128])
classThree = np.empty([1,128])
classFour = np.empty([1,128])
classFive = np.empty([1,128])
classSix = np.empty([1,128])
classSeven = np.empty([1,128])
classEight = np.empty([1,128])
classNine = np.empty([1,128])

for x in range (21000):
    if labelsOfUnlabeled[x]==0:
        classZero = np.concatenate((classZero, np.reshape(X2[x,],[1,128])), axis=0)
    elif labelsOfUnlabeled[x]==1:
        classOne = np.concatenate((classOne, np.reshape(X2[x,],[1,128])), axis=0)
    elif labelsOfUnlabeled[x]==2:
        classTwo = np.concatenate((classTwo, np.reshape(X2[x,],[1,128])), axis=0)
    elif labelsOfUnlabeled[x]==3:
        classThree = np.concatenate((classThree, np.reshape(X2[x,],[1,128])), axis=0)
    elif labelsOfUnlabeled[x]==4:
        classFour = np.concatenate((classFour, np.reshape(X2[x,], [1, 128])), axis=0)
    elif labelsOfUnlabeled[x]==5:
        classFive = np.concatenate((classFive, np.reshape(X2[x,], [1, 128])), axis=0)
    elif labelsOfUnlabeled[x]==6:
        classSix = np.concatenate((classSix, np.reshape(X2[x,], [1, 128])), axis=0)
    elif labelsOfUnlabeled[x]==7:
        classSeven = np.concatenate((classSeven, np.reshape(X2[x,], [1, 128])), axis=0)
    elif labelsOfUnlabeled[x]==8:
        classEight = np.concatenate((classEight, np.reshape(X2[x,], [1, 128])), axis=0)
    elif labelsOfUnlabeled[x]==9:
        classNine = np.concatenate((classNine, np.reshape(X2[x,], [1, 128])), axis=0)