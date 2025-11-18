import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint

# 1. Load Dataset
df = pd.read_csv(r"E:\Education\college\internship\ds_ibm\Attention_Span_Prediction_Project\preprocessing\final_dataset_preprocessed.csv")

print("Missing values:\n", df.isnull().sum())

X = df.drop("Time_Spent", axis=1)
y_raw = df["Time_Spent"].values.reshape(-1, 1)


# 2. Scale the Target Variable
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y_raw).ravel()  # Flatten for sklearn compatibility

# Save the target scaler
joblib.dump(scaler_y, "scaler_y.pkl")
print(" Target scaler saved as 'scaler_y.pkl'")


# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 4. Preprocessing Pipeline
categorical_cols = ['Content_Type', 'Time_Of_Day', 'Day_Of_Week', 'Device_Type', 'Platform_Group']

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regression', RandomForestRegressor(random_state=42))
])


# 5. Hyperparameter Tuning
param_dist = {
    'regression__n_estimators': randint(100, 500),
    'regression__max_depth': [None, 5, 10, 20, 30],
    'regression__min_samples_split': randint(2, 10),
    'regression__min_samples_leaf': randint(1, 5),
    'regression__criterion': ['squared_error', 'absolute_error']
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1,
    verbose=2
)


# 6. Train Model
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_


# 7. Evaluate
y_pred_scaled = best_model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1))

print("\n Best Parameters:", random_search.best_params_)
print("MSE:", mean_squared_error(y_test_real, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test_real, y_pred)))
print("MAE:", mean_absolute_error(y_test_real, y_pred))
print("R2 Score:", r2_score(y_test_real, y_pred))

# Cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
print("\nCross-Validated R2 Scores:", cv_scores)
print("Mean CV R2:", cv_scores.mean())


# 8. Plot CV Scores
plt.boxplot(cv_scores)
plt.title('Cross-Validated R2 Scores')
plt.ylabel('R2 Score')
plt.show()


# 9. Save Model
joblib.dump(best_model, 'model_dump.pkl')
print("\n Model saved as 'model_dump.pkl'")
