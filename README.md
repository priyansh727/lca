# LCA - Linear Regression on California Housing Dataset

**Subject:** Business Machine Learning (BML)

## Objective
Develop, train, and evaluate **Simple Linear Regression (SLR)** and
**Multiple Linear Regression (MLR)** models using the California Housing Dataset.

## Dataset
- Source: `sklearn.datasets.fetch_california_housing`
- Rows: 20,640
- Features (8): `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`,
  `AveOccup`, `Latitude`, `Longitude`
- Target: `MedHouseVal` (median house value in $100,000s)

## Files
- `california_housing_regression.py` - full pipeline: EDA, SLR, MLR, comparison, plots.

## How to Run
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
python california_housing_regression.py
```

## Pipeline
1. Load dataset and inspect (shape, stats, missing values).
2. EDA - correlation heatmap; auto-select strongest predictor for SLR.
3. **Simple Linear Regression** using the top-correlated feature (`MedInc`).
4. **Multiple Linear Regression** using all 8 features (with `StandardScaler`).
5. Evaluate both models on the test set using **MSE, RMSE, MAE, R^2**.
6. Compare SLR vs MLR and plot Actual-vs-Predicted + residuals.

## Metrics Used
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (R^2)

## Result Summary
Multiple Linear Regression outperforms Simple Linear Regression because it
uses more explanatory variables, resulting in a higher R^2 and lower error.
