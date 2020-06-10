import numpy as np
import pandas as pd

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
    del[df['Ticket']]
    del[df['Name']]
    df['Embarked'] = np.where(df['Embarked'] == 'C', 1, 0 )
    return df