#%%
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

print(os.listdir())

# %%
train = pd.read_csv("train.csv")

# %%
sns.distplot(train['Age'])
plt.show()

# %%
train.head()

# %%
def pipeline(df) :
    df['under_5'] = np.where(df['Age'] >= 5,1,0)
    df['Pclass1'] = np.where(df['Pclass'] == 1,1,0)
    df['cabinLetter'] = df['Cabin'].str[:1]
    aa = [] 
    cabinLetters = ['B','D','E','F']
    for row in df['cabinLetter'] :
        if row in cabinLetters :
            aa.append(1)
        else:
            aa.append(0)
    df['cabinLetter'] = aa
    del[df['Cabin']]
    df['Embarked'] = np.where(df['Embarked'] == 'C', 1, 0 )
    return df


# %%
df = pipeline(train)

# %%
train[(train['Survived'] == 1)]['under_5'].value_counts()

# %%
for value in train.Pclass.unique() :
   x =  train[(train['Survived'] == 1) & (train['Pclass'] == value)].shape[0] / train.shape[0]
   print(value, str(x))

# %%
plt.figure(figsize = (10,8))
sns.catplot(x = 'Survived', y="Fare",
            kind="box", dodge=False, data=df)

# %%
train.to_csv('train_cleansed.csv', index = False)

# %%
