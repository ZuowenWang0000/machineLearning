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
from sklearn.semi_supervised import label_propagation
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix


train_labeled = pd.read_hdf("train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")

train_labeled = np.array(train_labeled)

features = train_labeled [:, 1:129]

lables = train_labeled[:,0]

features = sklearn.preprocessing.scale(features)
train_unlabeled = sklearn.preprocessing.scale(np.array(train_unlabeled))

X = features
y = lables

X = np.concatenate((X,train_unlabeled),axis=0)
for i in range(21000):
    y = np.concatenate((y,np.array([-1])),axis = 0 )

slotSize = 2100

for i in range (int(21000/slotSize)):
    lp = LabelSpreading(kernel="knn", gamma=20, n_neighbors=15, alpha=0.05, max_iter=100, tol=0.001, n_jobs=1)
    lp.fit(X, y)

