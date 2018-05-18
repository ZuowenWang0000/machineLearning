

from matplotlib.ticker import NullFormatter
print(__doc__)
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from sklearn.decomposition import PCA
import sklearn


train_labeled = pd.read_hdf("train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")

train_labeled = np.array(train_labeled)

features = train_labeled [:, 1:129]

lables = train_labeled[:,0]

print("scaled features")
X = sklearn.preprocessing.scale(features[:9000,])
y = lables[:9000,]

print("pca to 2")
pca = PCA(n_components=2)
X = pca.fit_transform(X)

n_samples, n_features = X.shape
n_neighbors = 30


(fig, subplots) = plt.subplots(3, 5, figsize=(15, 8))
perplexities = [5,10,20,50,100]

red = y == 0
green = y == 1
black = y == 2
yellow = y == 3
orange = y == 4
green = y == 5
purple = y == 6
grey = y == 7
teal = y == 8
cyan = y == 9


# ax = subplots[0][0]
# ax.scatter(X[red, 0], X[red, 1], X[red,2],c="r")
# ax.scatter(X[green, 0], X[green, 1],X[green,2], c="g")
#
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# ax.zaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')
from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# tsne = manifold.TSNE(n_components=n_components, init='random',
#                          random_state=0, perplexity=100)
# # Y = tsne.fit_transform(X)
Y = X
#
# #######PLOTTING CONFIGURATIONS###############################
#
#
# ax.scatter(Y[red, 0], Y[red, 1],Y[red,2], c="r")
# ax.scatter(Y[green, 0], Y[green, 1], Y[green,2],c="g")
# ax.scatter(Y[black,0],Y[black,1],Y[black,2],c = "b")
# ax.scatter(Y[yellow,0],Y[yellow,1],Y[yellow,2],c = "y")
# ax.scatter(Y[orange,0],Y[orange,1],Y[orange,2],c = "orange")
# ax.scatter(Y[green,0],Y[green,1],Y[green,2],c = "green")
# ax.scatter(Y[purple,0],Y[purple,1],Y[purple,2],c = "purple")
# ax.scatter(Y[grey,0],Y[grey,1],Y[grey,2],c = "grey")
# ax.scatter(Y[teal,0],Y[teal,1],Y[teal,2],c = "teal")
# ax.scatter(Y[cyan,0],Y[cyan,1],Y[cyan,2], c = "cyan")
#
# ###############################################################
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# #
print("PCA50")
print("tsne100")

plt.show()

#
# for i, perplexity in enumerate(perplexities):
#     ax = subplots[0][i + 1]
#
#     t0 = time()
#     print(t0)
#     tsne = manifold.TSNE(n_components=n_components, init='random',
#                          random_state=0, perplexity=perplexity)
#     Y = tsne.fit_transform(X)
#     t1 = time()
#     print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
#     ax.set_title("Perplexity=%d" % perplexity)
#     ax.scatter(Y[red, 0], Y[red, 1],Y[red,2], c="r")
#     ax.scatter(Y[green, 0], Y[green, 1], Y[green,2],c="g")
#     ###############################
#     # ax.scatter(Y[black,0],Y[black,1],Y[black,2],c = "b")
#     # ax.scatter(Y[yellow,0],Y[yellow,1],Y[yellow,2],c = "y")
#     # ax.scatter(Y[orange,0],Y[orange,1],c = "orange")
#     # ax.scatter(Y[green,0],Y[green,1],c = "green")
#     # ax.scatter(Y[purple,0],Y[purple,1],c = "purple")
#     # ax.scatter(Y[grey,0],Y[grey,1],c = "grey")
#     # ax.scatter(Y[teal,0],Y[teal,1],c = "teal")
#     # ax.scatter(Y[cyan,0],Y[cyan,1], c = "cyan")
#
#
#     #############################
#     ax.xaxis.set_major_formatter(NullFormatter())
#     ax.yaxis.set_major_formatter(NullFormatter())
#     ax.zaxis.set_major_formatter(NullFormatter())
#     ax.axis('tight')
#
# plt.show()
fig = plt.figure()
ax = fig.add_subplot(111)
# tsne = manifold.TSNE(n_components=n_components, init='random',
#                          random_state=0, perplexity=100)
# Y = tsne.fit_transform(X)
Y = X

#######PLOTTING CONFIGURATIONS###############################


ax.scatter(Y[red, 0], Y[red, 1], c="r")
# ax.scatter(Y[green, 0], Y[green, 1],c="g")
# ax.scatter(Y[black,0],Y[black,1],c = "b")
# ax.scatter(Y[yellow,0],Y[yellow,1],c = "y")
# ax.scatter(Y[orange,0],Y[orange,1],c = "orange")
# ax.scatter(Y[green,0],Y[green,1],c = "green")
# ax.scatter(Y[purple,0],Y[purple,1],c = "purple")
# ax.scatter(Y[grey,0],Y[grey,1],c = "grey")
# ax.scatter(Y[teal,0],Y[teal,1],c = "teal")
# ax.scatter(Y[cyan,0],Y[cyan,1], c = "cyan")
# ax.scatter(Y[red, 2], Y[red, 1], c="r")
# ax.scatter(Y[green, 2], Y[green, 1],c="g")
# ax.scatter(Y[black,2],Y[black,1],c = "b")
# ax.scatter(Y[yellow,2],Y[yellow,1],c = "y")
# ax.scatter(Y[orange,2],Y[orange,1],c = "orange")
# ax.scatter(Y[green,2],Y[green,1],c = "green")
# ax.scatter(Y[purple,2],Y[purple,1],c = "purple")
# ax.scatter(Y[grey,2],Y[grey,1],c = "grey")
# ax.scatter(Y[teal,2],Y[teal,1],c = "teal")
# ax.scatter(Y[cyan,2],Y[cyan,1], c = "cyan")

###############################################################

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')


print("PCA50")
print("tsne100")

plt.show()