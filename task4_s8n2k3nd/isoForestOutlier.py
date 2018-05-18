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
import sklearn.naive_bayes as nb
import sklearn.ensemble as en


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

features = sklearn.preprocessing.scale(features)
train_unlabeled = sklearn.preprocessing.scale(np.array(train_unlabeled))

# gnb = nb.MultinomialNB()
# gnb.fit(features,lables)
#
# yresult = gnb.predict(train_unlabeled)
# np.savetxt('gaussianNB.csv',yresult,delimiter=',')
validate = np.loadtxt(open('benchmark2100.csv'),delimiter = ",")
valiStored = validate
train_unlabeledStored = train_unlabeled
# print(accuracy_score(validate,yresult))

iso = en.IsolationForest(n_estimators=100, max_samples='auto', contamination=0.3, max_features=128, bootstrap=False, n_jobs=-1, random_state=None, verbose=2)

iso.fit(train_unlabeled,validate)
truthTable = iso.predict(train_unlabeled)

inlier = []
inlierLabel = []
count = 0
for i in range(21000):
    if truthTable[i] == 1:
        temp = train_unlabeledStored[i,]
        inlier = np.append(inlier,temp)
        inlierLabel = np.append(inlierLabel,valiStored[i])

inlier = np.reshape(inlier,[int(len(inlier)/128), 128])
all_features = np.concatenate((features,inlier),axis=0)
all_labels = np.concatenate((lables,inlierLabel),axis=0)
print(all_features.shape)
print(all_labels.shape)

nn = neural_network.MLPClassifier(hidden_layer_sizes=(1024,1024, ),activation= 'relu', solver='adam', alpha=1,
learning_rate='adaptive', learning_rate_init=0.0001, power_t=0.5, shuffle=True,
    random_state=None, tol=0.00000001, verbose=True, max_iter = 400, early_stopping=False,momentum=0.9,
nesterovs_momentum=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

nn.fit(all_features,all_labels)
Validate9000 = nn.predict(features)
print(accuracy_score(lables,Validate9000))

print("100 estimators with rate0.3,two layers 1024")