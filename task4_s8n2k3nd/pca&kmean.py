from matplotlib.ticker import NullFormatter
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.decomposition import PCA
from sklearn import neural_network
from sklearn.metrics import accuracy_score
import sklearn


train_labeled = pd.read_hdf("train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")

train_labeled = np.array(train_labeled)
features = train_labeled [:, 1:129]
lables = train_labeled[:,0]


X = features
y = lables
X2 = train_unlabeled

##SCALLING DATA##
print("feature scaled")
X = sklearn.preprocessing.scale(X)
X2 =sklearn.preprocessing.scale(X2)

pca = PCA(n_components=3)
print("pca to 3 dimensions")
X = pca.fit_transform(X)

##pca the none-scaled unlabeled features to 50 dimensions
pca2 = PCA(n_components=3)
X2 = pca2.fit_transform(X2)

