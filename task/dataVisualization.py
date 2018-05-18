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
print("################")
print(features.shape)
print(train_unlabeled.shape)
all_data = np.concatenate((features,train_unlabeled),axis=0)


print (all_data.shape)
print (all_data)

pca = PCA(n_components=2, copy=True, whiten=False, svd_solver="auto", tol=0.0, iterated_power="auto", random_state=None)
visible = pca.fit_transform(features[:9000,])

principalDf = pd.DataFrame(data = visible
             , columns = ['principal component 1', 'principal component 2'])
lableXX = pd.DataFrame(lables[:50,],columns = ['lable'])
finalDf = pd.concat([principalDf, lableXX], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 Component PCA', fontsize = 20)

targets = ['0', '1', '2','3','4','5','6','7','8','9']
colors = ['r', 'g', 'b','c','m','y','k','w',]
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['lable'] == lableXX
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

plt.show()

print("PCA explained ratio")
print(pca.explained_variance_ratio_)

#
# #
# nn = neural_network.MLPClassifier(hidden_layer_sizes=(1024,1024,1024,),activation= 'relu', solver='adam', alpha=1,
# learning_rate='constant', learning_rate_init=0.00001, power_t=0.5, shuffle=True,
#     random_state=None, tol=0.0001, verbose=False, momentum=0.9,
# nesterovs_momentum=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# #
# # print ("running")
#
# #
# # nn.fit(all_data, all_lables)
# nn.fit(features, lables)
# # ##seems bigger alpha + smaller learning rate + appropriate numbers of layers & units => good fitting
# #
# # trainResult = nn.predict(all_data)
# # acc = accuracy_score(all_lables, trainResult)
# # print("acc1 alldata")
# # print(acc)
# # acc3 = accuracy_score(lables, trainResult[:9000,])
# # print("all data trained but accuracy on labled set")
# # print(acc3)
#
# lablesOfUnlabeled = nn.predict(train_unlabeled)
# all_lables = np.concatenate((lables,lablesOfUnlabeled),axis=0)
#
# nn2 = neural_network.MLPClassifier(hidden_layer_sizes=(1024,1024,1024,1024,),activation= 'relu', solver='adam', alpha=1,
# learning_rate='constant', learning_rate_init=0.00001, power_t=0.5, shuffle=True,
#     random_state=None, tol=0.0001, verbose=False, momentum=0.9,
# nesterovs_momentum=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#
# nn2.fit(all_data, all_lables)
#
#
#
# test = sklearn.preprocessing.scale(test)
# result = nn2.predict(test)
# print (result)
# np.savetxt('Guoye.csv',result,delimiter = ",")
#
# print("terminated")