"""
Taken and adapted from free version of SL course on Udacity
"""

from numpy import *
import pylab as pl
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from collections import defaultdict


d = defaultdict(LabelEncoder)
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
# Load the boston dataset and seperate it into training and testing set
iris = datasets.load_iris()
#X, y = df[columns], df['Walc']
X, y = iris.data, iris.target
offset = int(0.8 * len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# We will vary the depth of decision trees from 2 to 25
max_depth = arange(1, 6)
train_score = zeros(len(max_depth))
test_score = zeros(len(max_depth))

for i, d in enumerate(max_depth):
    estimator = DecisionTreeClassifier(max_depth=d)
    # Setup a Decision Tree Regressor so that it learns a tree with depth d
    # Fit the learner to the training data
    estimator.fit(X_train, y_train)

    # Find the score on the training set
    train_score[i] = accuracy_score(y_train, estimator.predict(X_train))
    # Find the score on the testing set
    test_score[i] = accuracy_score(y_test, estimator.predict(X_test))

pl.figure()
pl.ylim(ymax=2)
pl.title('Decision Trees: Performance vs Max Depth')
pl.plot(max_depth, test_score, label='Iris test score')
pl.plot(max_depth, train_score,label='Iris training score')

X, y = df[columns], df['Walc']
offset = int(0.9 * len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

max_depth = arange(2,25)
alc_train_score = zeros(len(max_depth))
alc_test_score = zeros(len(max_depth))

for i, d in enumerate(max_depth):
    estimator = DecisionTreeClassifier(max_depth=d)
    # Setup a Decision Tree Regressor so that it learns a tree with depth d
    # Fit the learner to the training data
    estimator.fit(X_train, y_train)

    # Find the score on the training set
    alc_train_score[i] = accuracy_score(y_train, estimator.predict(X_train))
    # Find the score on the testing set
    alc_test_score[i] = accuracy_score(y_test, estimator.predict(X_test))

# Plot training and test error as a function of the depth of the decision tree learnt
#lw=2

pl.plot(max_depth, alc_test_score, label='Alcohol test score')
pl.plot(max_depth, alc_train_score,label='Alcohol training score')

pl.legend()
pl.xlabel('Max Depth')
pl.ylabel('Score')
pl.show()