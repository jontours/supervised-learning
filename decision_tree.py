# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd

dataframe = pd.read_csv(filepath_or_buffer="~/Documents/repos/supervised_learning/student/student-mat.csv", sep=';')

clf = tree.DecisionTreeClassifier()

data = np.array(dataframe)
iris = load_iris()

print(data)



