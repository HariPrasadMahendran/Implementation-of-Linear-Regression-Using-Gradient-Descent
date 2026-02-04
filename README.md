# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

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
Hari Prasad M(25013933)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data (no header=None)
data = pd.read_csv(r"C:\Users\acer\Downloads\DATASET-20260131\50_Startups.csv")

# Use R&D Spend vs Profit for simple linear regression
x_data = data.iloc[:, 0].values   # R&D Spend
y_data = data.iloc[:, 4].values   # Profit

plt.scatter(x_data, y_data)
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Profit Prediction")

def computeCost(x, y, theta):
    m = len(y)
    h = x.dot(theta)
    return (1/(2*m)) * np.sum((h - y)**2)

data_n = data.values
m = len(y_data)

# Prepare x and y
x = np.c_[np.ones(m), x_data]
y = y_data.reshape(m, 1)
theta = np.zeros((2,1))

def gradientDescent(x, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        predictions = x.dot(theta)
        error = x.T.dot(predictions - y)
        theta -= (alpha / m) * error
        J_history.append(computeCost(x, y, theta))

    return theta, J_history

theta, J_history = gradientDescent(x, y, theta, 0.01, 1500)

print("h(x) = {} + {}x".format(round(theta[0,0],2), round(theta[1,0],2)))

# Plot Cost function
plt.figure()
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("Cost J(θ)")
plt.title("Cost Function")

# Plot regression line
plt.figure()
plt.scatter(x_data, y_data)
x_range = np.linspace(min(x_data), max(x_data), 100)
y_pred = theta[0][0] + theta[1][0] * x_range
plt.plot(x_range, y_pred, color='r')
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Profit Prediction")
plt.show()
/*
Program to implement the linear regression using gradient descent.
Developed by: 
RegisterNumber:  
*/
```

## Output:
<img width="856" height="568" alt="ML EXP 3(1)" src="https://github.com/user-attachments/assets/0fa98cd8-c212-4e97-ab61-bf74ab2d90d5" />
<img width="790" height="605" alt="ML EXP 3(2)" src="https://github.com/user-attachments/assets/590440ad-7539-4bf3-95ee-4ae61e40c89d" />
<img width="759" height="590" alt="ML EXP 3(3)" src="https://github.com/user-attachments/assets/9677a321-3268-4960-89bc-facfa562f851" />





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
