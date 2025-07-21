import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
import shap
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
tqdm.pandas()

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print("Train shape:", train.shape)
print("Test shape:", test.shape)
test_ID = test['Id']
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

y = train['SalePrice']
train.drop('SalePrice', axis=1, inplace=True)
full_data = pd.concat([train, test], axis=0)
full_data = pd.get_dummies(full_data)
imputer = SimpleImputer(strategy='median')
full_data_imputed = pd.DataFrame(imputer.fit_transform(full_data), columns=full_data.columns)
X = full_data_imputed[:len(y)]
X_test_final = full_data_imputed[len(y):]

xgb = XGBRegressor(
    n_estimators=1000, learning_rate=0.05, max_depth=3,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
cat = CatBoostRegressor(
    iterations=1000, learning_rate=0.05, depth=3,
    verbose=0, random_seed=42
)
stack = StackingRegressor(
    estimators=[('xgb', xgb), ('cat', cat)],
    final_estimator=XGBRegressor(n_estimators=500, learning_rate=0.01, random_state=42)
)

def rmse_cv(model, X, y):
    rmse = -cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=5)
    return rmse
print("XGBoost CV RMSE:", rmse_cv(xgb, X, y).mean())
print("CatBoost CV RMSE:", rmse_cv(cat, X, y).mean())
print("Stacking CV RMSE:", rmse_cv(stack, X, y).mean())

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
stack.fit(X_train, y_train)
y_pred = stack.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
print(f"Validation RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")

explainer = shap.Explainer(stack.namedestimators['xgb'])
shap_values = explainer(X_val[:100])
shap.plots.beeswarm(shap_values)

final_preds = stack.predict(X_test_final)
submission = pd.DataFrame({'Id': test_ID, 'SalePrice': final_preds})
submission.to_csv("submission.csv", index=False)
print("Submission file saved")