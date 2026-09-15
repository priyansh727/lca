"""
Simple & Multiple Linear Regression - California Housing Dataset
Subject: BML
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame
df.rename(columns={"MedHouseVal": "Price"}, inplace=True)

print("Shape:", df.shape)
print(df.head())

# 2. Split features (X) and target (y)
X = df.drop("Price", axis=1)
y = df["Price"]

# ============================================================
# SIMPLE LINEAR REGRESSION (using only MedInc)
# ============================================================
print("\n--- Simple Linear Regression ---")

X1 = df[["MedInc"]]

X_train, X_test, y_train, y_test = train_test_split(
    X1, y, test_size=0.2, random_state=42
)

slr = LinearRegression()
slr.fit(X_train, y_train)
y_pred = slr.predict(X_test)

print("Intercept :", slr.intercept_)
print("Slope     :", slr.coef_[0])
print("MSE  :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE  :", mean_absolute_error(y_test, y_pred))
print("R2   :", r2_score(y_test, y_pred))

# Plot
plt.scatter(X_test, y_test, alpha=0.3, label="Actual")
plt.plot(X_test, y_pred, color="red", label="Predicted")
plt.xlabel("MedInc")
plt.ylabel("Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.savefig("slr_plot.png")
plt.show()

# ============================================================
# MULTIPLE LINEAR REGRESSION (using all 8 features)
# ============================================================
print("\n--- Multiple Linear Regression ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlr = LinearRegression()
mlr.fit(X_train, y_train)
y_pred = mlr.predict(X_test)

print("Intercept    :", mlr.intercept_)
print("Coefficients :", mlr.coef_)
print("MSE  :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE  :", mean_absolute_error(y_test, y_pred))
print("R2   :", r2_score(y_test, y_pred))

# Plot
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Linear Regression")
plt.savefig("mlr_plot.png")
plt.show()
