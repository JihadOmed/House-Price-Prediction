# house_price_prediction_advanced.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
import shap
import optuna
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
tqdm.pandas()

# -------------------------------
# 1. Load Data
# -------------------------------
print("📥 Loading datasets...")
train = pd.read_csv(r'C:\Users\HC\Desktop\House Price\train.csv')
test = pd.read_csv(r'C:\Users\HC\Desktop\House Price\test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

test_ID = test['Id']
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)

# -------------------------------
# 2. Target and Combine Data
# -------------------------------
y = train['SalePrice']
train.drop(['SalePrice'], axis=1, inplace=True)
full_data = pd.concat([train, test], axis=0).reset_index(drop=True)

# -------------------------------
# 3. Feature Engineering
# -------------------------------

print("🔧 Feature engineering...")

# Fix skewed numeric features by log-transform
numeric_feats = full_data.select_dtypes(include=[np.number]).columns
skewness = full_data[numeric_feats].apply(lambda x: x.skew()).sort_values(ascending=False)
skewed_feats = skewness[abs(skewness) > 0.75].index

for feat in skewed_feats:
    # Add 1 to avoid log(0)
    full_data[feat] = np.log1p(full_data[feat])

# Fill missing categorical features with 'Missing'
cat_feats = full_data.select_dtypes(include=['object']).columns
full_data[cat_feats] = full_data[cat_feats].fillna('Missing')

# For numerical features, fill missing with median later in pipeline

# Interaction feature example: total area
full_data['TotalSF'] = full_data['TotalBsmtSF'] + full_data['1stFlrSF'] + full_data['2ndFlrSF']

# Drop features with too many missing values (>50%)
missing_percent = full_data.isnull().mean()
drop_features = missing_percent[missing_percent > 0.5].index
full_data.drop(drop_features, axis=1, inplace=True)

# -------------------------------
# 4. Prepare Preprocessing Pipeline
# -------------------------------

print("⚙️ Preparing preprocessing pipelines...")

numeric_features = full_data.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = full_data.select_dtypes(include=['object']).columns.tolist()

# Imputer for numeric features - median
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Encoder for categorical features - OneHot
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # fix here
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# -------------------------------
# 5. Prepare Training Data
# -------------------------------

print("🚦 Fitting and transforming data...")

X_all = preprocessor.fit_transform(full_data)

# Split back to train/test
X_train = X_all[:len(y), :]
X_test = X_all[len(y):, :]

# -------------------------------
# 6. Hyperparameter Optimization (Optuna)
# -------------------------------

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'random_state': 42,
        'tree_method': 'gpu_hist' if 'cuda' in XGBRegressor().get_params() else 'hist',
        'verbosity': 0
    }
    model = XGBRegressor(**params)
    scores = cross_val_score(model, X_train, y, cv=5, scoring='neg_root_mean_squared_error')
    return -scores.mean()

def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 2000),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_seed': 42,
        'verbose': 0
    }
    model = CatBoostRegressor(**params)
    scores = cross_val_score(model, X_train, y, cv=5, scoring='neg_root_mean_squared_error')
    return -scores.mean()

print("🔍 Running hyperparameter optimization for XGBoost...")
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=True)

print("🔍 Running hyperparameter optimization for CatBoost...")
study_cat = optuna.create_study(direction='minimize')
study_cat.optimize(objective_cat, n_trials=30, show_progress_bar=True)

# -------------------------------
# 7. Train Optimized Models
# -------------------------------

print("🚀 Training models with optimized hyperparameters...")

best_xgb_params = study_xgb.best_params
best_cat_params = study_cat.best_params

xgb_model = XGBRegressor(**best_xgb_params)
cat_model = CatBoostRegressor(**best_cat_params)

# Split data for validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42)

# Train with early stopping
xgb_model.fit(X_tr, y_tr,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=50,
              verbose=False)

cat_model.fit(X_tr, y_tr,
              eval_set=(X_val, y_val),
              early_stopping_rounds=50,
              verbose=False)

# -------------------------------
# 8. Stacking Ensemble
# -------------------------------

print("🔗 Creating stacking ensemble...")

stack_model = StackingRegressor(
    estimators=[('xgb', xgb_model), ('cat', cat_model)],
    final_estimator=Ridge(alpha=1.0)
)
stack_model.fit(X_tr, y_tr)

# Validation predictions
y_pred = stack_model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

print(f"Validation RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")

# -------------------------------
# 9. SHAP Explainability
# -------------------------------
print("🔍 Generating SHAP values...")

explainer_xgb = shap.Explainer(xgb_model)
shap_values_xgb = explainer_xgb(X_val[:100])
shap.plots.beeswarm(shap_values_xgb)

explainer_cat = shap.Explainer(cat_model)
shap_values_cat = explainer_cat(X_val[:100])
shap.plots.beeswarm(shap_values_cat)

# -------------------------------
# 10. Final Prediction and Submission
# -------------------------------

print("📈 Predicting on test data...")
final_predictions = stack_model.predict(X_test)

submission = pd.DataFrame({'Id': test_ID, 'SalePrice': np.expm1(final_predictions)})  # inverse log1p transform

submission.to_csv("submission_advanced.csv", index=False)
print("✅ Submission saved as 'submission_advanced.csv'")
