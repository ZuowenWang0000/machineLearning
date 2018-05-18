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


features = sklearn.preprocessing.scale(features)
train_unlabeled = sklearn.preprocessing.scale(np.array(train_unlabeled))

nn = neural_network.MLPClassifier(hidden_layer_sizes=(1024, 1024, ),activation= 'relu', solver='adam', alpha=1,
learning_rate='adaptive', learning_rate_init=0.0001, power_t=0.5, shuffle=True,
    random_state=None, tol=0.00000001, verbose=True, max_iter = 400, early_stopping=False,momentum=0.9,
nesterovs_momentum=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

nn.fit(features, lables)
np.savetxt('benchmark2100.csv',nn.predict(train_unlabeled),delimiter=',')


