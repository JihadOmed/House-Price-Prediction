# House Price Prediction - Advanced Regression Techniques

This repository contains a comprehensive machine learning project for predicting house prices using the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset from Kaggle.

---

## Project Overview

Accurately predicting house prices is a classic regression problem with wide applications in real estate valuation, investment analysis, and economic forecasting. This project builds and evaluates state-of-the-art regression models, including XGBoost, CatBoost, and their stacking ensemble, to predict sale prices based on various housing features.

---

## Dataset

The dataset contains 1460 training samples and 1459 test samples with 80+ features describing houses in Ames, Iowa. These features include numerical and categorical data such as:

- Lot size, year built, overall quality, neighborhood
- Number of rooms, garage size, basement condition, etc.

The test set lacks the target variable `SalePrice`, which the models predict.

---

## Technical Approach

### Data Preprocessing
- Combined train and test datasets for consistent feature engineering.
- Converted categorical variables to dummy/one-hot encoded variables.
- Used median imputation to fill missing values across features.
- Dropped identifier columns (`Id`) and separated the target variable (`SalePrice`).

### Models Used
- **XGBoost Regressor**: Gradient boosting framework with tuned hyperparameters.
- **CatBoost Regressor**: Gradient boosting with native handling of categorical features.
- **Stacking Regressor**: Ensemble model stacking XGBoost and CatBoost as base learners, with another XGBoost as the final estimator to improve generalization.

### Model Evaluation
- Used 5-fold cross-validation with RMSE as the scoring metric.
- Trained on 80% of the data and validated on 20%.
- Performance metrics on the validation set:
  - Root Mean Squared Error (RMSE)
  - R² Score
  - Mean Absolute Error (MAE)
  - Mean Absolute Percentage Error (MAPE)

### Model Explainability
- Used SHAP (SHapley Additive exPlanations) to interpret feature importance and model predictions visually.

---

## Results

| Model           | CV RMSE       | Validation RMSE | R² Score | MAE        | MAPE      |
|-----------------|---------------|-----------------|----------|------------|-----------|
| XGBoost         | 25,681.71     | -               | -        | -          | -         |
| CatBoost        | 27,267.62     | -               | -        | -          | -         |
| Stacking Ensemble | 28,246.26    | 31,391.81       | 0.8715   | 17,448.78  | 9.96 %    |

The stacking ensemble showed strong predictive performance and robust generalization.

---

## Usage

1. Clone this repository.
2. Install dependencies (e.g., via `pip install -r requirements.txt`).
3. Place the `train.csv` and `test.csv` files from the Kaggle dataset in the working directory.
4. Run the main script to preprocess data, train models, evaluate performance, generate SHAP plots, and create the submission CSV file.

---

## Libraries & Tools

- Python 3.x
- pandas, numpy: Data manipulation
- matplotlib, seaborn: Visualization
- scikit-learn: Model selection, metrics, preprocessing
- xgboost: Gradient boosting implementation
- catboost: Gradient boosting with categorical support
- shap: Model explainability
- tqdm: Progress visualization

---

## References

- Kaggle Dataset: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- XGBoost: https://xgboost.readthedocs.io/
- CatBoost: https://catboost.ai/
- SHAP: https://shap.readthedocs.io/

---

## Author

Jihad - Data Analyst passionate about building robust predictive models and deriving actionable insights from data.

---

## License

This project is open-source and available under the MIT License.

---

*Feel free to explore the code, reproduce the results, and improve the models further!*

