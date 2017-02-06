# Import the necessary modules and libraries
import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from plot_learning_curve import plot_learning_curve
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from collections import defaultdict
d = defaultdict(LabelEncoder)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

dataframe = pd.read_csv(filepath_or_buffer="~/Documents/repos/supervised_learning/student/student-por.csv", sep=';')
dataframe1 = pd.read_csv(filepath_or_buffer="~/Documents/repos/supervised_learning/student/student-mat.csv", sep=';')
dfs = [dataframe, dataframe1]
df = pd.concat(dfs)
df = df.drop_duplicates()
injector = np.copy(df)
#df = df.apply(lambda x: d[x.name].fit_transform(x))


cols_to_transform = ['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
                     'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
dummy_frame = pd.get_dummies(df, columns = cols_to_transform )

columns = dummy_frame.columns.tolist()
columns = [c for c in columns if c not in ["Walc",'Dalc', 'G1', 'G2', 'G3', 'Fedu', 'Medu', 'studytime', 'famrel']]
#print(df.corr()['Walc'])
#print(df.values.shape)
target = 'Walc'
# Generate the training set.  Set random_state to be able to replicate results.
train, test = train_test_split(df, test_size = 0.1, random_state=42)
digits = load_digits()
X, y = digits.data, digits.target


title = "Learning Curves (Decision Tree)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

plot_learning_curve(tree.DecisionTreeClassifier(), title, dummy_frame[columns], dummy_frame[target], ylim=(0.2, 1.01), cv=cv, n_jobs=4)
print(test.values.shape)
print(train.values.shape)

title = "Learning Curves (Random Forrest)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.1, random_state=0)
#df[columns], df[target],
estimator = RandomForestClassifier(n_estimators=19)
plot = plot_learning_curve(estimator, title, dummy_frame[columns], dummy_frame[target], ylim=(0.2, 1.01), cv=cv, n_jobs=4)
#'Mjob_at_home', 'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Mjob_health', 'Mjob_at_home']
goout = ['goout','age', 'sex_F', 'sex_M', 'Mjob_at_home', 'Mjob_other', 'Mjob_services', 'Mjob_teacher', 'Mjob_health', 'Mjob_at_home']
estimator = MLPClassifier(learning_rate='invscaling', activation='logistic', hidden_layer_sizes=(5,),
                          learning_rate_init=0.001, alpha=.01, max_iter=100, random_state=1,
       solver='sgd', tol=0.0001, verbose=True,
       warm_start=False)

cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)


plot = plot_learning_curve(estimator, "Learning Curves (Neural Net)", dummy_frame[goout], dummy_frame[target], ylim=(0.2, 1.01), cv=cv, n_jobs=4)
plot.show()

clf = tree.DecisionTreeClassifier(random_state=42, max_depth=20)
clf.fit(train[columns], train[target])
predictions = clf.predict(test[columns])
print(clf.score(test[columns], test[target]))

neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(train[columns], train[target])
print(neigh.score(test[columns], test[target]))

cv = ShuffleSplit(n_splits=100, test_size=0.1, random_state=0)
plot = plot_learning_curve(KNeighborsClassifier(n_neighbors=2), title, df[columns], df[target], ylim=(0.2, 1.01), cv=cv, n_jobs=4)
plot.show()

n_net = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(2,), random_state=1)
#train[columns], train[target]
n_net.fit(
    df[columns], df[target]
)
print('Nueral net')
print(n_net.score(test[columns], test[target]))
forest = RandomForestClassifier(n_estimators=19, random_state=42)

forest.fit(train[columns], train[target])
print('Forest')
print(forest.score(test[columns], test[target]))

with open("alcohol.dot.tree", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)






