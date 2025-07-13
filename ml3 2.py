import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
plt.scatter(X_test[:, 0], y_test, color='royalblue', label='Actual')
plt.scatter(X_test[:, 0], y_pred, color='lightgreen', label='Predicted')
plt.xlabel('Feature 1')
plt.ylabel('Target')
plt.title('Linear Regression on IRIS Dataset')
plt.legend()
plt.show()
