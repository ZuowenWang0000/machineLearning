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
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.learning_curve import validation_curve

train_labeled = pd.read_hdf("train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")
benchmark8000 = np.loadtxt('HugoBenchMark.csv', delimiter = ',')

train_labeled = np.array(train_labeled)

features = train_labeled [:, 1:129]

lables = train_labeled[:,0]
trainingSetLabels = lables

X = sklearn.preprocessing.scale(features)
y = lables

params_range =[(512,),(1024,),(2048,),(256,256,256,256,256,256,256,),(500)]

clf1 = neural_network.MLPClassifier(hidden_layer_sizes=(256, ),activation= 'relu', solver='adam', alpha=0.1,
learning_rate='constant', learning_rate_init=0.001, power_t=0.5, shuffle=True,
    random_state=None, tol=0.00001, verbose=True, momentum=0.9,
nesterovs_momentum=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

train_scores, test_scores = validation_curve(estimator = clf1, X = X, y = lables, param_name = 'hidden_layer_sizes', param_range=params_range, cv = 10)

train_mean = np.mean(train_scores, axis= 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std=np.std(test_scores, axis = 1)

params_range = [1,2,3,4,5]

plt.plot(params_range, train_mean,color = 'blue', marker = 'o',markersize = 5,label = 'training accuracy')

plt.fill_between(params_range, train_mean + train_std, train_mean - train_std, alpha = 0.15)

plt.plot(params_range, test_mean, color = 'green', linestyle = '--', marker = 's', markersize = 5, label = 'validation accuracy')

plt.fill_between(params_range, test_mean + test_std, test_mean-test_std, alpha = 0.15, color = 'green')

plt.grid()
plt.xscale('log')
plt.legend(loc = 'lower right')
plt.xlabel('net')
plt.ylabel('accuracy')
plt.ylim([0.8,1.0])
plt.show()







