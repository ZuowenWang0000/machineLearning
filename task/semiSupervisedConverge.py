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


rng = np.random.RandomState(0)
# indices = np.arange(len(digits.data))
# rng.shuffle(indices)

X = features
y = lables

X = np.concatenate((X,train_unlabeled),axis=0)
for i in range(21000):
    y = np.concatenate((y,np.array([-1])),axis = 0 )

n_total_samples = 30000
n_labeled_points = 9000

unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]



max_iterations = 10
step = 2100 ##we delete 5 most unreliable labels each round

for i in range(max_iterations):
    print("round: {}.".format(i))
    if len(unlabeled_indices) == 0:
        print("No unlabeled items left to label.")
        break
    y_train = y
    print("using label spreading")
    lp_model = LabelSpreading(kernel="knn", gamma=20, n_neighbors=15, alpha=0.05, max_iter=100, tol=0.001, n_jobs=1)
    lp_model.fit(X, y_train)

    predicted_labels = lp_model.transduction_[unlabeled_indices]
    true_labels = y[unlabeled_indices]

    cm = confusion_matrix(true_labels, predicted_labels,
                          labels=lp_model.classes_)

    print("Iteration %i %s" % (i, 70 * "_"))
    print("Label Spreading model: %d labeled & %d unlabeled (%d total)"
          % (n_labeled_points, n_total_samples - n_labeled_points,
             n_total_samples))

    print(classification_report(true_labels, predicted_labels))

    print("Confusion matrix")
    print(cm)

    # compute the entropies of transduced label distributions
    pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)

    # select up to 5 digit examples that the classifier is most uncertain about
    mostcertain_index = np.argsort(pred_entropies)[::1]
    mostcertain_index = mostcertain_index[np.in1d(mostcertain_index, unlabeled_indices)][:step]

    # keep track of indices that we get labels for
    delete_indices = np.array([])

    # # Visualize the gain only on the first 5
    # k = 0
    # kmax = 5
    # f.text(.05, (max_iterations - i - 0.5) / max_iterations,
    #        "model %d\n\nfit with\n%d labels" % ((i + 1), n_labeled_points),
    #        size=8)
    #
    # for index, image_index in enumerate(uncertainty_index):
    #     image = images[image_index]
    #     sub = f.add_subplot(max_iterations, kmax, index + 1 + (kmax * i))
    #     sub.imshow(image, cmap=plt.cm.gray_r, interpolation='none')
    #     sub.set_title("predict: %i\ntrue: %i" % (lp_model.transduction_[image_index], y[image_index]), size=10)
    #     sub.axis('off')
    #     k += 1
    #     if k == kmax:
    #         break

    # labeling points, remote from labeled set
    # delete_index, = np.where(unlabeled_indices == image_index)
    delete_indices = np.concatenate((delete_indices, uncertainty_index))
    print("delete_indices:{}".format(delete_indices))
    print("delete_indices shape:{}".format(delete_indices.shape))

    print("Before Deleting:")
    print("unlabeled_indices:{}".format(unlabeled_indices))
    print("unlabeled_indices shape:{}".format(unlabeled_indices.shape))
    unlabeled_indices = np.delete(unlabeled_indices, delete_indices)

    print("unlabeled_indices:{}".format(unlabeled_indices))
    print("unlabeled_indices shape:{}".format(unlabeled_indices.shape))
    n_labeled_points += len(uncertainty_index)

    print("n_labeled_points: {}".format(n_labeled_points))

print(n_labeled_points)
np.savetxt('roundsPseudoLabel.csv',predicted_labels,delimiter = ',')


