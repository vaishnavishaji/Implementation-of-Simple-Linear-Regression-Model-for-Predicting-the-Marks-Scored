# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.

2.Set variables for assigning dataset values. 

3.Import linear regression from sklearn. 

4.Assign the points for representing in the graph. 

5.Predict the regression for marks by using the representation of the graph. 

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VAISHNAVI S.A
RegisterNumber:  212223220119
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
Dataset

![dataset](https://github.com/vaishnavishaji/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151444759/8475a77e-526c-498c-86d9-bf6cc5370460)

Head Values

![head](https://github.com/vaishnavishaji/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151444759/64d079f4-41b8-4559-8925-d4c46a750f2f)

Tail Values

![tail](https://github.com/vaishnavishaji/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151444759/e3fa9404-6691-4019-bd49-652abe3e04de)

X and Y values

![xyvalues](https://github.com/vaishnavishaji/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151444759/84d22c9e-9624-47b7-8f55-26b4c34d3af9)

Predication values of X and Y

![predict ](https://github.com/vaishnavishaji/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151444759/85d2bf91-0230-4ec7-86ff-dd8087b0294c)

MSE,MAE and RMSE

![values](https://github.com/vaishnavishaji/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151444759/0dc41180-e4ea-43e7-8758-13777831d824)

Training Set

![train](https://github.com/vaishnavishaji/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151444759/f691506d-789e-4612-8727-de36da233bc4)

Testing Set

![test](https://github.com/vaishnavishaji/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151444759/3c80d423-45fe-4eca-9380-5737751b9a75)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
