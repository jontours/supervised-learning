# Import the necessary modules and libraries
import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

dataframe = pd.read_csv(filepath_or_buffer="~/Documents/repos/supervised_learning/student/student-mat.csv", sep=';')
cols_to_transform = ['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
                     'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic' ]
dataframe = pd.get_dummies(dataframe, columns = cols_to_transform )

columns = dataframe.columns.tolist()
target = 'G3'
# Generate the training set.  Set random_state to be able to replicate results.
train = dataframe.sample(frac=0.8, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = dataframe.loc[~dataframe.index.isin(train.index)]
print(train.shape)
print(test.shape)
clf = tree.DecisionTreeClassifier()
clf.fit(train[columns], train[target])
predictions = clf.predict(test[columns])
print(test.shape)
print(clf.score(test, test[target]))

neigh = KNeighborsClassifier(n_neighbors=8)
neigh.fit(train[columns], train[target])
print(neigh.score(test, test[target]))

with open("alcohol.dot.tree", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)






