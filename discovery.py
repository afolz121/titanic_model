#%%
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import re
# %%
train = pd.read_csv("train.csv")

# %%
sns.distplot(train['Age'])
plt.show()
# %%
train.head()

# %%
train[(train['Survived'] == 1)]['under_5'].value_counts()

# %%
for value in train.Pclass.unique() :
   x =  train[(train['Survived'] == 1) & (train['Pclass'] == value)].shape[0] / train.shape[0]
   print(value, str(x))

# %%
plt.figure(figsize = (10,8))
sns.catplot(x = 'Survived', y="Fare",
            kind="box", dodge=False, data=train)

# %%
train.head()
# %%
import re

s = 'asdf=5;iwantthis123jasd'
result = re.search('asdf=5;(.*)123jasd', s)
print(result.group(1))

s = 'Braund, Mr. Owen Harris'
result = re.search(",\\s*([^.]*)", s)
print(result.group(1))

# %%
aa = []
for row in train['Name'] :
    result = re.search(",\\s*([^.]*)", row)
    aa.append(result.group(1))

train['surname'] = aa

# %%
