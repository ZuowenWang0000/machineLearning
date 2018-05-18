import pandas as pd
import numpy as np
import sklearn
from sklearn import neural_network
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
import pylab
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TkAgg')

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

km = KMeans(n_clusters=10, init= a , n_init=1, max_iter=300, tol=0.0001,precompute_distances="auto", verbose=0, random_state=None,
            copy_x=True, n_jobs=-1, algorithm="auto").fit(train_unlabeled)
print(km.cluster_centers_)
print(km.labels_)

all_lables = np.concatenate((lables,km.labels_),axis=0)
# print("################")
# print(features.shape)
# print(train_unlabeled.shape)
# all_data = np.concatenate((features,train_unlabeled),axis=0)
#
#
# print (all_data.shape)
# print (all_data)

pca = PCA(n_components=3, copy=True, whiten=False, svd_solver="auto", tol=0.0, iterated_power="auto", random_state=None)
visible = pca.fit_transform(features[:200,])

principalDf = pd.DataFrame(data = visible
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
lableXX = pd.DataFrame(lables[:200,],columns = ['lable'])
finalDf = pd.concat([principalDf, lableXX], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Z Label')
# ax.set_zlable('Principal Component 3', fontsize = 15)

ax.set_title('3 Component PCA', fontsize = 20)

targets = [0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]
colors = ['r', 'g', 'b','c','m','y','k','w','crimson','plum']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['lable'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               ,finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 25)
ax.legend(targets)
ax.grid()

pylab.show()
#
# print("PCA explained ratio")
# print(pca.explained_variance_ratio_)


#
nn = neural_network.MLPClassifier(hidden_layer_sizes=(1024,),activation= 'relu', solver='adam', alpha=0.1,
learning_rate='constant', learning_rate_init=0.0001, power_t=0.5, shuffle=True,
    random_state=None, tol=0.0001, verbose=False, momentum=0.9,
nesterovs_momentum=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#
# print ("running")

#
# nn.fit(all_data, all_lables)
# # nn.fit(features, lables)
# # ##seems bigger alpha + smaller learning rate + appropriate numbers of layers & units => good fitting
# #
# trainResult = nn.predict(all_data)
# acc = accuracy_score(all_lables, trainResult)
# print("acc1 alldata")
# print(acc)
# acc3 = accuracy_score(lables, trainResult[:9000,])
# print("all data trained but accuracy on labled set")
# print(acc3)
#
# #
# features = sklearn.preprocessing.scale(features)
# nn.fit(features, lables)
# trainResult2 = nn.predict(features)
# acc2 = accuracy_score(lables, trainResult2)
#
# print("acc2")
# print(acc2)
#
# print("terminated")