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
train_unlabeled = np.array(train_unlabeled)

features = train_labeled [:, 1:129]

lables = train_labeled[:,0]

print(lables)
print(features)
print(lables.size)
print(features.size)

# voted = np.loadtxt(open('votedLabels.csv'),delimiter = ",")
#
# filteredVotedData =np.empty([1,128])
# filteredVotedLabel = np.empty([])
# for i in range(21000):
#     print(i)
#     if voted[i] != -1:
#         filteredVotedData = np.concatenate((filteredVotedData,np.reshape(train_unlabeled[i,],[1,128])),axis = 0)
#         filteredVotedLabel = np.append(filteredVotedLabel,voted[i])


filteredVotedData = np.loadtxt(open('filteredVotedData.csv'),delimiter = ",")
filteredVotedLabel = np.loadtxt(open('filteredVotedLabels.csv'),delimiter = ",")

all_data = np.concatenate((features,filteredVotedData),axis = 0)
all_labels = np.concatenate((lables,filteredVotedLabel),axis=0)

all_data = sklearn.preprocessing.scale(all_data)


nn = neural_network.MLPClassifier(hidden_layer_sizes=(1024,1024,1024,1024, ),activation= 'relu', solver='adam', alpha=1,
learning_rate='adaptive', learning_rate_init=0.0001, power_t=0.5, shuffle=True,
    random_state=None, tol=0.0000001, verbose=True, max_iter = 400, early_stopping=False,momentum=0.9,
nesterovs_momentum=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


nn.fit(all_data,all_labels)
trainingSetResult = nn.predict(sklearn.preprocessing.scale(features))
print(accuracy_score(lables,trainingSetResult))