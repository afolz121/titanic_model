import numpy as np
import pandas as pd
import re

def min_max(col) :
    minimum = np.min(col)
    maximum = np.max(col)
    aa = []
    for row in col :
        aa.append((row - minimum) / (maximum - minimum))
    return aa

def pipeline(df) :
    df['under_5'] = np.where(df['Age'] >= 5,1,0)
    df['Pclass1'] = np.where(df['Pclass'] == 1,1,0)
    del[df['Cabin']]
    del[df['Ticket']]
    #del[df['Name']]
    df['Embarked'] = np.where(df['Embarked'] == 'S', 1, 0 )
    target = df['Survived'].copy()
    del[df['Survived']]
    df['Sex'] = np.where(df['Sex'] == 'male', 1, 0)
    del[df['PassengerId']]
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    aa = []
    for row in df['Name'] :
        result = re.search(",\\s*([^.]*)", row)
        aa.append(result.group(1))
    bb= []
    for row in df['Name'] :
        bb.append(row.split(' ', 2)[1])
    df['title'] = bb
    special_titles = ['Master.','nada','Mrs.']
    df['special_titles'] = np.where(df['title'].isin(special_titles), 1, 0)
    del[df['title']]
    del[df['Name']]
    df['fsize'] = df['Parch'] + df['SibSp']
    del[df['fsize'], df['SibSp']]
    for col in df.columns :
        df[col] = min_max(df[col])

    
    return df, target