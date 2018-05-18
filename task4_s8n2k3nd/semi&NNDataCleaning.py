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


train_unlabeled = sklearn.preprocessing.scale(train_unlabeled)
features = sklearn.preprocessing.scale(features)

lp = LabelSpreading(kernel='knn', gamma=20, n_neighbors=7, alpha=0.2, max_iter=50, tol=0.01, n_jobs=-1)

y = lables
for i in range(21000):
    y = np.concatenate((y,np.array([-1])),axis = 0 )

all_data = np.concatenate((features,train_unlabeled),axis=0)

lp.fit(all_data,y)
Yresult = lp.predict(all_data)
print(lp.score(all_data, Yresult))

np.savetxt('semiLabelsOfUnlabeled2.csv',Yresult,delimiter = ",")