import pandas as pd
import numpy as np
import sklearn
from sklearn import neural_network
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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


cataOne = 0


for x in range (0,8999):
    if train_labeled[x,0]==0:
        a[0] = a[0]+ features[x,]
        cataOne = np.append(cataOne, features[x,])
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


print(cataOne.shape)
cataOne = np.reshape(cataOne[1:110849:1],[866,128])
print(cataOne.shape)

slice = cataOne[:, :9:1]
print(slice.shape)

# f1 = plt.figure(1)
# plt.subplot(111)
#
# Xcor = np.arange(0,866,1)
# plt.scatter(Xcor,slice[:,1])
#
# f1.show()
#
# f2 = plt.figure(1)
# plt.subplot(111)
# plt.scatter(slice[:,0], slice[:,1])
# f2.show()

pca = PCA(n_components=2, copy=True, whiten=False, svd_solver="auto", tol=0.0, iterated_power="auto", random_state=None)
visible = pca.fit_transform(train_labeled [:, 1:129][:140,])

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(visible[:,0], visible[:,1], 0, zdir='y', s=30, c=None, depthshade=True)
#
# plt.show()

f2 = plt.figure(1)
plt.subplot(111)
plt.scatter(visible[:,0], visible[:,1])
f2.show()