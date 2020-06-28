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

data.head()

# %%
dtree = DecisionTreeClassifier(max_depth= 4)
dtree_fit = dtree.fit(data, target)

dtree_predict = dtree_fit.predict(data)

# %%
print(classification_report(target, dtree_predict))

# %%
forest = RandomForestClassifier(max_depth= 3, n_estimators= 100)
forest_fit = forest.fit(data, target)
forest_predict = forest_fit.predict(data)

# %%
print(classification_report(target, forest_predict))

# %%
svmach = SVC(kernel= 'rbf', degree = 2, C = 5)
svmach_fit = svmach.fit(data, target)
svmarch_predict = svmach_fit.predict(data)

# %%
print(classification_report(target, svmarch_predict))

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
