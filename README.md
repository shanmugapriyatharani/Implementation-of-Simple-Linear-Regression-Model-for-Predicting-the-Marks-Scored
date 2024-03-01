# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KALPANA S
RegisterNumber: 212222040069

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('/content/ml - Sheet1.csv')
df.head(10)

plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')

x=df.iloc[:,0:1]
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train,Y_train)

X_train
Y_train

lr.predict(X_test.iloc[0].values.reshape(1,1))

plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')
*/  
*/
```

## Output:

![image](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393427/55e5a5f2-43b4-4f89-b152-8b6324e8f3b3)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
