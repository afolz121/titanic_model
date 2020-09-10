#%%
import numpy as np
import pandas as pd
import os
import sys
from cleanse_pipe import cleanse_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
import pycaret
from pycaret.classification import *

#%%
train = pd.read_csv('train.csv')
# %%
train.head()
# %%
data, passengers = cleanse_data(train)
# %%
data.head()
# %%
exp_clf = setup(data, target = 'Survived', train_size = .7, normalize= True, log_experiment= False)
# %%

exp_clf[0].head()

#%%

best = compare_models()
# %%
catboost = create_model('gbc')

# %%
tuned_cat = tune_model(catboost, n_iter= 200)
# %%

test = pd.read_csv('test.csv')

test_data, passengers1 = cleanse_data(test)

preds = predict_model(tuned_cat, data = test_data)

ready_preds = pd.DataFrame(passengers1, columns = ['PassengerId'])
ready_preds['Survived'] = preds['Label']

ready_preds.to_csv('titanic_preds.csv', index = False)

# %%
