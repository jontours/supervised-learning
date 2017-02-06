from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from plot_learning_curve import plot_learning_curve
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from collections import defaultdict
import seaborn as sns
import pandas as pd
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

plt = sns.pairplot(data, hue="target", size=2.5)
plt.savefig("Iris.png")


title = "Learning Curves (Decision Tree) max_depth=3"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = tree.DecisionTreeClassifier(max_depth=3)
plt = plot_learning_curve(estimator, title, X, y, ylim=(0.2, 1.01), cv=cv, n_jobs=4)

title = "Learning Curves (SVM, RBF kernel, C=35, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(kernel='rbf', gamma=0.001, C=35)
plot_learning_curve(estimator, title, X, y, (0.2, 1.01), cv=cv, n_jobs=4)
d = 13

#hidden_layer_sizes=(2,4,5,6,4,3),
estimator = MLPClassifier(learning_rate='constant', activation='tanh', hidden_layer_sizes=(d, d-4, d-8),
                              learning_rate_init=0.01, max_iter=50, random_state=1,
                              solver='lbfgs', tol=0.00001, verbose=True,
                              warm_start=False)

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

plot_learning_curve(estimator, "Learning Curves (Neural Net)", X, y, ylim=(0.01, 1.01), cv=cv, n_jobs=4)

estimator = AdaBoostClassifier(n_estimators=4, random_state=1)

plot_learning_curve(estimator, "Learning Curves (AdaBoost with 4 learners) Iris Dataset", X, y, ylim=(0.2, 1.01), cv=cv, n_jobs=4)

estimator = KNeighborsClassifier(n_neighbors=8)

plot_learning_curve(estimator, "Learning Curves (KNN with 8 neighbors) Iris Dataset", X, y, ylim=(0.2, 1.01), cv=cv, n_jobs=4)

#plt.show()