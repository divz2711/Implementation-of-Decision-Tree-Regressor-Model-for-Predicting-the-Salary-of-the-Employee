# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.calculate Mean square error,data prediction and r2.
 
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Divya S
RegisterNumber: 212221040042
*/

import pandas as pd
data=pd.read_csv("Salary.csv")

data.head()

data.info()


data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()


y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
## data.head()
![image](https://github.com/divz2711/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121245222/2543a6e4-84df-447b-98cb-71bbcdd77112)

## data.info
![image](https://github.com/divz2711/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121245222/38408a8b-4cc2-4a33-9798-9178a455e1d1)

## isnull() and sum()
![image](https://github.com/divz2711/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121245222/fac6c1a8-fbec-47fe-8601-81aebe5d7dc5)

## data.head() for salary
![image](https://github.com/divz2711/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121245222/0b54814a-7435-45b9-b143-32423efdef37)

## MSE value
![image](https://github.com/divz2711/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121245222/91e6f78d-181a-4085-8cc9-e34161eb8185)

## r2 value
![image](https://github.com/divz2711/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121245222/00a3ec20-0a93-4bad-bd70-ddb61b1fa5eb)

## data prediction
![image](https://github.com/divz2711/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/121245222/dc6343da-58bb-4593-a5a9-1765cfbe339c)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
