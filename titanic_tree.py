# %%
import numpy as np
import pandas as pd
from sklearn import tree
import graphviz
import pydotplus
import collections
from IPython.display import Image
from cleanse_pipe import cleanse_data
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score


#%%
train = pd.read_csv('train.csv')
data, passengers = cleanse_data(train)

target = data['Survived']
del[data['Survived']]

#%% 
# run decision tree model

dtree = tree.DecisionTreeClassifier(max_depth = 4)
dtree_fit = dtree.fit(data, target)
dtree_predict = dtree_fit.predict(data)


#%% 
dot_data = tree.export_graphviz(dtree_fit,
                                feature_names=data.columns,
                                out_file=None,
                                filled=True,
                                rounded=True)
 
graph = pydotplus.graph_from_dot_data(dot_data)
 
colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)
 
for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))
 
for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph = pydotplus.graph_from_dot_data(dot_data)  

Image(graph.create_png())


# %%
print(classification_report(target, dtree_predict))

print(roc_auc_score(target, dtree_predict))

#%%
pd.DataFrame(confusion_matrix(target,dtree_predict), columns = ['No Survive Actual','Survive Actual'], 
index = ['No Survive Predicted','Survive Predicted'])
# %%
test = pd.read_csv('test.csv')

test1, test_passengers = cleanse_data(test)
test_preds = dtree_fit.predict(test1)

preds_df = pd.DataFrame(test_passengers, columns = ['PassengerId'])
preds_df['Survived'] = test_preds

preds_df.to_csv("titanic_preds.csv", header = True, index = False)

# %%
