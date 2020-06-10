#%%
import numpy as np
import pandas as pd
import os
import sys
from cleanse_pipe import pipeline

# %%
train = pd.read_csv('train.csv')

# %%
train = pipeline(train)

# %%
train.head()

# %%
