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

train_labeled = pd.read_hdf("train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")

train_labeled = np.array(train_labeled)

features = train_labeled [:, 1:129]

lables = train_labeled[:,0]
trainingSetLabels = lables

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


count0 = 0
count1 = 0
count9 = 0
for x in range (0,8999):
    if train_labeled[x,0]==0:
        a[0] = a[0]+ features[x,]
        count0 = count0 +1
    elif train_labeled[x,0]==1:
        a[1] = a[1]+ features[x,]
        count1 = count1 +1
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
        count9 = count9 +1

print("count0")
print(count0)
print("count1")
print(count1)
print(count9)

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

lp = LabelSpreading(kernel="knn", gamma=20, n_neighbors=10, alpha=0.05, max_iter=1000, tol=0.001, n_jobs=-1)
for i in range(21000):
    lables = np.append(lables,-1)

all_data = np.concatenate((features,train_unlabeled),axis=0)
lp.fit(all_data, lables)

all_labels = lp.transduction_

print("all_labels shape")
print(all_labels.shape)


nn = neural_network.MLPClassifier(hidden_layer_sizes=(1024, ),activation= 'relu', solver='adam', alpha=0.1,
learning_rate='constant', learning_rate_init=0.001, power_t=0.5, shuffle=True,
    random_state=None, tol=0.0001, verbose=True, momentum=0.9,
nesterovs_momentum=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

nn.fit(all_data,all_labels)

print("ACC on training set")
print(accuracy_score(lables,nn.predict(features)))

test = sklearn.preprocessing.scale(test)
result = nn.predict(test)
np.savetxt('knnemiSurpervisedNNalpha0.05.csv', result,delimiter = ",")

print("alpha = 0.05")
# all_lables = np.concatenate((lables,labelsOfUnlabeled),axis=0)
# all_data = np.concatenate((features,train_unlabeled),axis=0)
#
# print("***************************************")
# print (all_data.shape)
# print (all_lables.shape)
#
# hist,bins = np.histogram(all_lables, bins = [-0.1,0.1,1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1,10.1])
# print (hist)
# print (bins)
#
# test = sklearn.preprocessing.scale(test)

# np.savetxt('clustercluster.csv',km2.labels_,delimiter = ",")
