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

train.head()

# %%
train.groupby('Survived')['Sex'].value_counts(normalize = True)

# %%
male = train[train['Sex'] == 'male']

# %%
for num in [0,1] :
    male[male['Survived'] == num]['Age'].hist(bins = 20)
    plt.title(str(num))
    plt.show()

# %%
male.head()

# %%
aa= []
for row in male['Name'] :
    aa.append(row.split(' ', 2)[1])

# %%
male['title'] = aa

# %%
male.groupby('Survived')['title'].value_counts(normalize = True)

# %%
