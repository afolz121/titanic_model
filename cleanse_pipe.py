#%%
import numpy as np
import pandas as pd
import re
pd.set_option('max_columns', 30)

def cleanse_data(df) :
    aa = []
    for name in df.Name :
        aa.append(re.findall(r"(?<=, )[\w]*",name)[0])
    df['Name'] = aa
    df['Name'] = np.where(df['Name'].isin(['Mlle','Mme','Lady','the']), 'Ms', df['Name'])
    df['Name'] = np.where(df['Name'].isin(['Master', 'Dr', 'Major', 'Sir']), 'Master', df['Name'])
    df['Name'] = np.where(df['Name'].isin(['Rev', 'Col','Jonkheer', 'Capt', 'Don','Dona']), 'Mr', df['Name'])
    for index, row in df.iterrows() :
        if np.isnan(row['Age']) :
            df['Age'][index] = df[df['Name'] == df['Name'][index]]['Age'].mean()
        else:
            pass
    df['Age'] = df['Age'].fillna(df['Age'].mean())

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix = 'Embarked')
    df = pd.concat([df, embarked_dummies], axis = 1)
    del[df['Embarked']]
    del[df['Cabin'], df['Ticket']]
    df['Alone'] = np.where((df['SibSp'] == 0) & (df['Parch'] == 0), 1, 0)
    name_dummy = pd.get_dummies(df['Name'], prefix = 'Title')
    df = pd.concat([df,name_dummy], axis = 1)
    df['Sex_Male'] = np.where(df['Sex'] == 'male', 1,0)
    passengers = df['PassengerId']
    del[df['PassengerId'], df['Sex'], df['Name']]
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    aa = []
    for row in df['Fare'] :
        if row in pd.Interval(left = -0.1, right = 7.0) :
            aa.append(1)
        elif row in pd.Interval(left = 7.0, right = 10.0) :
            aa.append(2)
        elif row in pd.Interval(left = 10, right = 21) :
            aa.append(3)
        elif row in pd.Interval(left = 21.0, right = 39.0) :
            aa.append(4)
        elif row > 39:
            aa.append(5)
        else:
            aa.append(9999)
        
    df['Farebin'] = aa

    aa = []
    for row in df['Age'] :
        if row in pd.Interval(left = -.01, right = 20) :
            aa.append(1)
        elif row in pd.Interval(left = 20, right = 26) :
            aa.append(2)
        elif row in pd.Interval(left = 26, right = 32) :
            aa.append(3)
        elif row in pd.Interval(left = 32, right = 38) :
            aa.append(4)
        elif row > 38 :
            aa.append(5)
        else:
            aa.append(9999)

    df['Agebin'] = aa

    return df, passengers
