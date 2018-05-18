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

nn = neural_network.MLPClassifier(hidden_layer_sizes=(256, ),activation= 'relu', solver='adam', alpha=1,
learning_rate='constant', learning_rate_init=0.0001, power_t=0.5, shuffle=True,
    random_state=None, tol=0.01, verbose=True, max_iter = 400, early_stopping=False,momentum=0.9,
nesterovs_momentum=True, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

featuresNormalized = sklearn.preprocessing.scale(features)
nn.fit(featuresNormalized,lables)
result = nn.predict(featuresNormalized)
print(accuracy_score(lables,result))

# unlabeledTrainFeaturesNormalized

testScaledFeatures = sklearn.preprocessing.scale(test)
UnlabeledScaledFeatures = sklearn.preprocessing.scale(train_unlabeled)
LabelsOfUnlabeled = nn.predict(UnlabeledScaledFeatures)
LogProbTrain = nn.predict_proba(UnlabeledScaledFeatures)

record = []
for index in range(21000):
    i = LabelsOfUnlabeled[index]
    if LogProbTrain[index,int(i)]>0.95:
        record.append(UnlabeledScaledFeatures[[index]])
        record.append(LabelsOfUnlabeled[index])