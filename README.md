# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: 
RegisterNumber:  
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term
    X = np.c_[np.ones(len(X)), X]

    # Initialize theta with zeros
    theta = np.zeros(X.shape[1]).reshape(-1, 1)

    # Perform gradient descent
    for _ in range(num_iters):
        # Calculate predictions
        predictions = X.dot(theta).reshape(-1, 1)

        # Calculate errors
        errors = (predictions - y).reshape(-1, 1)

        # Update theta using gradient descent
        theta -= learning_rate * (2 / len(X)) * X.T.dot(errors)

    return theta

# Read data from CSV file
data = pd.read_csv('/content/50_Startups.csv', header=None)

# Extract features (X) and target variable (y)
X = data.iloc[1:, :-2].values.astype(float)
y = data.iloc[1:, -1].values.reshape(-1, 1)

# Standardize features and target variable
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y)

# Example usage
# X_array = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
# y_array = np.array([2, 7, 11, 16])

# Learn model parameters
theta_result = linear_regression(X_scaled, y_scaled)

# Predict target value for a new data point
new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(-1, 1)
new_scaled = scaler.fit_transform(new_data)

prediction = np.dot(np.append(1, new_scaled), theta_result)
prediction = prediction.reshape(-1, 1)

# Inverse transform the prediction to get the original scale
predicted_value = scaler.inverse_transform(prediction)

print(f"Predicted value: {predicted_value}")
```
## Output:
![Screenshot 2024-03-14 222613](https://github.com/selva258963/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121961701/6723adaa-2550-4544-ab71-1e03b7cc44f3)
![Screenshot 2024-03-14 222624](https://github.com/selva258963/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/121961701/85397ef4-df01-4c31-a9dc-a3c0956a4def)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
