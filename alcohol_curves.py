# Import the necessary modules and libraries
import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from plot_learning_curve import plot_learning_curve
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from collections import defaultdict
import seaborn as sns
from seaborn.linearmodels import corrplot

d = defaultdict(LabelEncoder)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

dataframe = pd.read_csv(filepath_or_buffer="~/Documents/repos/supervised_learning/student/student-por.csv", sep=';')
dataframe1 = pd.read_csv(filepath_or_buffer="~/Documents/repos/supervised_learning/student/student-mat.csv", sep=';')
dfs = [dataframe, dataframe1]
df = pd.concat(dfs)
df = df.drop_duplicates()

df = df.apply(lambda x: d[x.name].fit_transform(x))

cols_to_transform = ['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
                     'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

columns = df.columns.tolist()
columns = [c for c in columns if c not in ["Walc",'Dalc', 'G1', 'G2', 'G3', 'Fedu', 'Medu', 'studytime', 'famrel','schoolsup',
                     'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']]
print(columns)
print(df.corr()['Walc'])
#print(df.values.shape)
target = 'Walc'

sns.pairplot(df, vars=['goout', 'sex'], hue="Walc", size=2.5)
#corrplot(df, annot=False)

train, test = train_test_split(df, test_size = 0.1, random_state=42)

title = "Learning Curves (Decision Tree) max_depth=17"

cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

plot_learning_curve(tree.DecisionTreeClassifier(max_depth=17), title, df[columns], df[target], ylim=(0.2, 1.01), cv=cv, n_jobs=4)
print(test.values.shape)
print(train.values.shape)
print(len(columns))
title = "Learning Curves (Random Forrest)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
estimator = RandomForestClassifier(n_estimators=19)
plot = plot_learning_curve(estimator, title, df[columns], df[target], ylim=(0.2, 1.01), cv=cv, n_jobs=4)

estimator = MLPClassifier(learning_rate='constant', activation='logistic',hidden_layer_sizes=(10,),
                          learning_rate_init=0.001, alpha=0.001, max_iter=200,
       solver='lbfgs', tol=0.0001, verbose=True,
       warm_start=False)

cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=0)


goout = ['goout', 'sex', 'age', 'Fjob']
plot = plot_learning_curve(estimator, "Learning Curves (Neural Net) alcohol one 10 node layer", df[goout], df[target], ylim=(0.2, 1.01), cv=cv, n_jobs=4)

clf = tree.DecisionTreeClassifier(random_state=42, max_depth=20)
clf.fit(train[columns], train[target])
predictions = clf.predict(test[columns])
print(clf.score(test[columns], test[target]))

neigh = KNeighborsClassifier(n_neighbors=11)
neigh.fit(train[columns], train[target])
print(neigh.score(test[columns], test[target]))

cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
plot_learning_curve(KNeighborsClassifier(n_neighbors=3), "KNN with 3 neighbors (alcohol data set)", df[columns], df[target], ylim=(0.2, 1.01), cv=cv, n_jobs=4)
plot.show()
title = "Learning Curves (SVM, Poly kernel, degree=4)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
estimator = SVC(kernel='poly', degree=4, coef0=10.0, random_state=1)
plot_learning_curve(estimator, title, df[columns], df[target], (0.2, 1.01), cv=cv, n_jobs=4)


n_net = MLPClassifier(solver='lbfgs', alpha=1e-5,
                      random_state=1)
n_net.fit(
    df[columns], df[target]
)
print('Nueral net')
print(n_net.score(test[columns], test[target]))
forest = RandomForestClassifier(n_estimators=19, random_state=42, max_depth=2)

boost = AdaBoostClassifier(n_estimators=45, random_state=1, learning_rate=0.75)
plot_learning_curve(boost, "Learning Curves (AdaBoost w/ 45 learners) Alcohol Dataset", df[columns], df[target], (0.2,1.01), cv=cv, n_jobs=4)
plot_learning_curve(forest, "Learning Curves (RandomForest w/ 19 estimators and max_depth=2) Alcohol Dataset", df[columns], df[target], (0.2,1.01), cv=cv, n_jobs=4)

forest.fit(train[columns], train[target])
print('Forest')
print(forest.score(test[columns], test[target]))
plot.show()

with open("alcohol.dot.tree", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)






