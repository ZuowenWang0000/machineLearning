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

n_total_samples = 30000
n_labeled_points = 9000

unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]

# for i in range (int(21000/slotSize)):
#     lp = LabelSpreading(kernel="knn", gamma=20, n_neighbors=15, alpha=0.05, max_iter=100, tol=0.1, n_jobs=-1)
#     lp.fit(X, y)
#     y = lp.transduction_
#     print(y)
#     pred_entropies = stats.distributions.entropy(lp.label_distributions_.T)
#
#     # select up to 5 digit examples that the classifier is most uncertain about
#     uncertain_index = np.argsort(pred_entropies)[::-1]
#     uncertain_index = uncertain_index[np.in1d(uncertain_index, unlabeled_indices)][:21000-(i+1)*slotSize]
#     print(uncertain_index)
#     print(uncertain_index.shape)
#
#     count = 0
#     for j in range(uncertain_index.shape[0]):
#         count = count + 1
#         ind = uncertain_index[j]
#         y[ind] = -1
#     print(count)
# np.savetxt('recursiveLabels.csv',y,delimiter = ',')

sanwan = np.loadtxt(open('recursiveLabels.csv'),delimiter = ',')

nn = neural_network.MLPClassifier(hidden_layer_sizes=(1024, ),activation= 'relu', solver='adam', alpha=1,
learning_rate='adaptive', learning_rate_init=0.001, power_t=0.5, shuffle=True,
    random_state=None, tol=0.0000001, verbose=True, max_iter = 400, early_stopping=True,validation_fraction=0.05,momentum=0.9,
nesterovs_momentum=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

nn.fit(X,sanwan)

validate = nn.predict(features)
print(accuracy_score(lables,validate))

testReulst = nn.predict(sklearn.preprocessing.scale(test))

np.savetxt('lastTry2.csv',testReulst,delimiter = ',')



