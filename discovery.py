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
train.head()

# %%
sns.distplot(train['Age'])
plt.show()

# %%
