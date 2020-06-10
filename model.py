#%%
import numpy as np
import pandas as pd
import os
import sys
from cleanse_pipe import pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC

# %%
train = pd.read_csv('train.csv')

# %%
data, target = pipeline(train)

# %%
dtree = DecisionTreeClassifier(max_depth= 4)
dtree_fit = dtree.fit(data, target)

dtree_predict = dtree_fit.predict(data)

# %%
print(classification_report(target, dtree_predict))

# %%
forest = RandomForestClassifier(max_depth= 4, n_estimators= 500)
forest_fit = forest.fit(data, target)
forest_predict = forest_fit.predict(data)

# %%
print(classification_report(target, forest_predict))

# %%
svmach = SVC(kernel= 'rbf', degree = 4)
svmach_fit = svmach.fit(data, target)
svmarch_predict = svmach_fit.predict(data)

# %%
print(classification_report(target, svmarch_predict))

# %%
