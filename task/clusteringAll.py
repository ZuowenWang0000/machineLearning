import pandas as pd
import numpy as np
import sklearn
from sklearn import neural_network
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelSpreading
from sklearn import datasets

train_labeled = pd.read_hdf("train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")

train_labeled = np.array(train_labeled)

features = train_labeled [:, 1:129]

lables = train_labeled[:,0]

print(lables)
print(features)
print(lables.size)
print(features.size)

##check if the data is balanced
hist,bins = np.histogram(lables, bins = [-0.1,0.1,1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1])
print (hist)
print (bins)

print(train_labeled.shape)
print(train_labeled[:,0])

# print("here")
# print(train_labeled[1,])
n = 10
m = 128
a = [0] * n
for i in range(n):
    a[i] = [0] * m



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

a = np.array(a)
a = sklearn.preprocessing.scale(a)
train_unlabeled = sklearn.preprocessing.scale(train_unlabeled)
features = sklearn.preprocessing.scale(features)

km = KMeans(n_clusters=10, init= a , n_init=1, max_iter=1000, tol=0.0001,precompute_distances="auto", verbose=0, random_state=None,
            copy_x=True, n_jobs=-1, algorithm="auto").fit(train_unlabeled)
print(km.cluster_centers_)
print(km.labels_)

all_lables = np.concatenate((lables,km.labels_),axis=0)
all_data = np.concatenate((features,train_unlabeled),axis=0)


print (all_data.shape)
print (all_data)

hist,bins = np.histogram(all_lables, bins = [-0.1,0.1,1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1])
print (hist)
print (bins)
#
label_prop_model = LabelSpreading(kernel="knn", gamma=20, n_neighbors=7, alpha=0.2, max_iter=30, tol=0.001, n_jobs=1)

rng = np.random.RandomState(42)
random_unlabeled_points = rng.rand(len(iris.target)) < 0.3
labels = np.copy(iris.target)
labels[random_unlabeled_points] = -1
label_prop_model.fit(iris.data, labels)

#
# km2 = KMeans(n_clusters=10, init= a , n_init=1, max_iter=1000, tol=0.0001,precompute_distances="auto", verbose=0, random_state=None,
#             copy_x=True, n_jobs=-1, algorithm="auto").fit(train_unlabeled)