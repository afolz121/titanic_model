#%%
import numpy as np
import pandas as pd
import os
import sys
from cleanse_pipe import pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# %%
train = pd.read_csv('train.csv')

# %%
data, target = pipeline(train)

data.head()

# %%
dtree = DecisionTreeClassifier(max_depth= 4)
dtree_fit = dtree.fit(data, target)

dtree_predict = dtree_fit.predict(data)

# %%
print(classification_report(target, dtree_predict))

# %%
forest = RandomForestClassifier(bootstrap=True, criterion='gini', max_depth= None, max_features = 'log2', min_samples_leaf= 5,
                                min_samples_split = 2, n_estimators= 200)
forest_fit = forest.fit(data, target)
forest_predict = forest_fit.predict(data)

#%%
rforest = RandomForestClassifier()

grid = {"max_depth": [None],
              "max_features": [3,"sqrt", "log2"],
              "min_samples_split": [n for n in range(1, 9)],
              "min_samples_leaf": [5, 7],
              "bootstrap": [False, True],
              "n_estimators" :[200, 500],
              "criterion": ["gini", "entropy"]}


rforest_grid = GridSearchCV(rforest, param_grid = grid, cv=10, scoring="roc_auc", n_jobs= -1, verbose = 1)

rforest_grid_fit = rforest_grid.fit(data, target)

print(rforest_grid_fit.best_params_)

# %%
print(classification_report(target, forest_predict))
print(roc_auc_score(target, forest_predict))

# %%
svmach = SVC(kernel= 'rbf', degree = 1, C = 10)
svmach_fit = svmach.fit(data, target)
svmarch_predict = svmach_fit.predict(data)

# %%
print(classification_report(target, svmarch_predict))

#%%
from sklearn.model_selection import GridSearchCV

params = {
    "C" : [0.1, .5, .8, 1, 2, 3, 10],
    "kernel" : ['rbf'] , 
    "degree" : [1,2,3,4,5]
}

grid = GridSearchCV(SVC(), param_grid= params, n_jobs= 5, cv = 20, error_score= "accuracy")

grid_fit = grid.fit(data, target)

grid_fit.best_params_

# %%
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(learning_rate = 0.1,
n_estimators = 200,
max_depth = 3)

gb_fit = gb.fit(data, target)
gb_predict = gb.predict(data)

print(classification_report(target, gb_predict))

# %%
from sklearn.model_selection import GridSearchCV

params = {'kernel' : ['rbf','poly','sigmoid'],
	'C' : [.1,.5,.8,1,2,10],
	'degree' : [1,2,3,4,5]
}

gs = GridSearchCV(SVC(), param_grid=params, scoring = 'accuracy', 
cv=10, n_jobs= -1
)

gs_fit = gs.fit(data, target)
print(gs_fit.best_params_)

# %%
test = pd.read_csv('test.csv')
test['Survived'] = 1

passengers = test['PassengerId'].copy()

test, target_dum = pipeline(test)
test_preds = forest_fit.predict(test)

preds_df = pd.DataFrame(passengers, columns = ['PassengerId'])
preds_df['Survived'] = test_preds

preds_df.to_csv("titanic_preds.csv", header = True, index = False)


# %%
# %%
