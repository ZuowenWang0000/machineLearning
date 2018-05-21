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

params_range =[10,50,200,500,1000,5000]

clf1 = RandomForestClassifier(n_estimators=800, criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)


train_scores, test_scores = validation_curve(estimator = clf1, X = X, y = lables, param_name = 'n_estimators', param_range=params_range, cv = 5)

train_mean = np.mean(train_scores, axis= 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std=np.std(test_scores, axis = 1)

# params_range = [1,2,3,4,5]

plt.plot(params_range, train_mean,color = 'blue', marker = 'o',markersize = 5,label = 'training accuracy')

plt.fill_between(params_range, train_mean + train_std, train_mean - train_std, alpha = 0.15)

plt.plot(params_range, test_mean, color = 'green', linestyle = '--', marker = 's', markersize = 5, label = 'validation accuracy')

plt.fill_between(params_range, test_mean + test_std, test_mean-test_std, alpha = 0.15, color = 'green')

plt.grid()
plt.xscale('log')
plt.legend(loc = 'lower right')
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.ylim([0.8,1.0])
plt.show()







