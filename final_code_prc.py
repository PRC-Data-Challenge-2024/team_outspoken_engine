# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.


# MODEL v9
# %%
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

# %%
# Load the dataset and create a 'route' feature by combining departure and arrival airport codes
url = "./comp_data_2024-10-20/challenge_set.csv"
raw_data = pd.read_csv(url)
raw_data['route'] = raw_data['adep'] + "-" + raw_data['ades']

# %%
# Display column names in the dataset for exploratory purposes
list(raw_data.columns)

# %%
# Define the predictor features and the target variable
pred_features = [
    'callsign',
    'adep',
    'country_code_adep',
    'ades',
    'country_code_ades',
    'aircraft_type',
    'wtc',
    'airline',
    'flight_duration',
    'taxiout_time',
    'flown_distance',
    'route',
]
X = raw_data[pred_features]  # Feature set
y = raw_data[['tow']]        # Target variable (Take-off weight, tow)

# %%
# Separate the features into categorical and numerical categories for preprocessing
categorical_features = [
    'callsign',
    'adep',
    'country_code_adep',
    'ades',
    'country_code_ades',
    'aircraft_type',
    'wtc',
    'airline',
    'route',
]
numerical_features = [
    'flight_duration',
    'taxiout_time',
    'flown_distance',
]

# %%
# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.12, random_state=123)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=123)

# %%
# Define preprocessing steps for numerical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  # Standardize numerical features
])

# Define preprocessing steps for categorical data
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# %%
# Combine preprocessing for numerical and categorical data using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# %%
# Define the model pipeline, combining preprocessing and an XGBoost regressor model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(objective='reg:squarederror', random_state=12345))
])

# %%
# Set up the parameter grid for hyperparameter tuning
param_grid = {
    'regressor__max_depth': [3, 6, 9],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__min_child_weight': [1, 3, 5],
    'regressor__subsample': [0.6, 0.8, 1.0],
    'regressor__colsample_bytree': [0.6, 0.8, 1.0]
}

# Perform grid search with cross-validation to identify the best model parameters
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

# %%
# Retrieve the best parameters from the grid search results
best_params = grid_search.best_params_

# Train the final model using the best parameters and validation data for early stopping
best_xgb_params = {param.replace('regressor__', ''): value for param, value in best_params.items()}
final_model = XGBRegressor(objective='reg:squarederror', random_state=1234, **best_xgb_params)

# %%
# Apply preprocessing transformations to training, validation, and test sets
X_train_processed = preprocessor.fit_transform(X_train)
X_valid_processed = preprocessor.transform(X_valid)
X_test_processed = preprocessor.transform(X_test)

# Fit the final model with early stopping to prevent overfitting
final_model.fit(
    X_train_processed,
    y_train,
    eval_set=[(X_valid_processed, y_valid)],
    verbose=True
)

# %%
# Make predictions on the test set using the final model
y_pred = final_model.predict(X_test_processed)

# %%
# Evaluate the model's performance on the test set
rmse_test = mean_squared_error(y_test, y_pred, squared=False)
print("Best Parameters:", best_params)
print("Root Mean Squared Error on Test Set:", rmse_test)

# %%
# Use the best model from grid search for predictions and evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("Best Parameters:", grid_search.best_params_)
print("Root Mean Squared Error on Test Set:", rmse)

# %% [markdown]
# # Predict take-off weight (tow) for submission dataset

# %%
# Load submission dataset and add the 'route' feature
subm_data = pd.read_csv("./comp_data_2024-10-20/final_submission_set.csv")
subm_data['route'] = subm_data['adep'] + "-" + subm_data['ades']

# %%
# Preview the submission data
subm_data.head()

# %%
# Extract prediction features from the submission data
Xs = subm_data[pred_features]

# %%
# Display columns of the submission feature set to verify
Xs.columns

# %%
# Make predictions for the submission data using the best model
ys = best_model.predict(Xs)

# %%
# Prepare a DataFrame for the submission results
df_pred = pd.DataFrame()
df_pred['flight_id'] = subm_data['flight_id']
df_pred['tow'] = ys

# %%
# Display the head of the submission predictions DataFrame
df_pred.head()

# %%
# Save the final model and the submission predictions
ver = 'v0'
model_path = ver + '_model.joblib'
joblib.dump(best_model, model_path)
csv_path = 'team_outspoken_engine_' + ver + '_796a128a-c833-453a-8653-5347905ae539.csv'
df_pred.to_csv(csv_path, index=False)



# MODEL v1 - Random Forest

# %%

# Import necessary libraries for data processing, model building, and evaluation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
import joblib

# %%
# Load the dataset from a specified path
url = "./comp_data_2024-10-20/challenge_set.csv"
raw_data = pd.read_csv(url)

# Create a new column 'route' by concatenating departure and arrival airports
raw_data['route'] = raw_data['adep'] + "-" + raw_data['ades']

# %%
# Display columns in the dataset for inspection
list(raw_data.columns)

# %%
# Select features for training and target variable
pred_features = [
    'callsign', 'adep', 'country_code_adep', 'ades', 'country_code_ades',
    'aircraft_type', 'wtc', 'airline', 'flight_duration', 'taxiout_time',
    'flown_distance', 'route'
]
X = raw_data[pred_features]  # Predictor features
y = raw_data[['tow']]        # Target variable: Take-off Weight (tow)

# %%
# Separate categorical and numerical features for preprocessing
categorical_features = [
    'callsign', 'adep', 'country_code_adep', 'ades', 'country_code_ades',
    'aircraft_type', 'wtc', 'airline', 'route'
]
numerical_features = ['flight_duration', 'taxiout_time', 'flown_distance']

# %%
# Split the dataset into training and testing sets (98% training, 2% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=123)

# %%
# Set up a preprocessing pipeline for categorical features: use OneHotEncoder to handle categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# %%
# Combine preprocessing steps for categorical features using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ]
)

# %%
# Create a pipeline combining preprocessing and model training with RandomForestRegressor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestRegressor(n_estimators=256, n_jobs=-1, verbose=True, random_state=12345))
])

# %%
# Train the Random Forest model with the training dataset
model.fit(X_train, y_train.values.ravel())

# %%
# Predict on the test set to evaluate the model's performance
y_pred = model.predict(X_test)

# %%
# Check the number of samples in the test set
len(X_test)

# %%
# Calculate the RMSE (Root Mean Squared Error) as an evaluation metric for model performance
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Test RMSE: {rmse}')

# %%
# Optionally, print feature importances for interpretability
# feature_importances = model.feature_importances_
# for feature, importance in zip(X_train.columns, feature_importances):
#     print(f'Feature: {feature}, Importance: {importance}')

# %% [markdown]
# ## Predict 'tow' for submission dataset

# %%
# Load final submission dataset and create 'route' column
subm_data = pd.read_csv("./comp_data_2024-10-20/final_submission_set.csv")
subm_data['route'] = subm_data['adep'] + "-" + subm_data['ades']

# %%
# Prepare the submission dataset with selected predictor features
Xs = subm_data[pred_features]

# %%
# Predict 'tow' for the final submission dataset using the trained model
ys = model.predict(Xs)

# %%
# Compile the prediction results in a new DataFrame for submission
df_pred = pd.DataFrame()
df_pred['flight_id'] = subm_data['flight_id']
df_pred['tow'] = ys

# %%
# Define versioning for model and submission file
ver = 'v1'

# Save the trained model using joblib for reuse
model_path = ver + '_model.joblib'
joblib.dump(model, model_path)

# Export the prediction results to CSV for submission
csv_path = 'team_outspoken_engine_' + ver + '_796a128a-c833-453a-8653-5347905ae539.csv'
df_pred.to_csv(csv_path, index=False)

# %%



# MODEL v2 - average v9 and v11

# %%
# Import the pandas library for data manipulation and analysis
import pandas as pd

# %%
# Define the path to the CSV file containing the raw data
url = "./comp_data_2024-10-20/challenge_set.csv"
# Load the dataset into a DataFrame
raw_data = pd.read_csv(url)

# Create a new column 'route' by concatenating the departure and destination airports
raw_data['route'] = raw_data['adep'] + "-" + raw_data['ades']

# Display the first few rows of the dataset to verify the data was loaded correctly
raw_data.head()

# %%
# Display the column names of the raw dataset
raw_data.columns

# %%
# Aggregate the data by route, airline, and aircraft type
# Calculate the count, mean, and standard deviation for the 'tow' (takeoff weight) column
agg_df = raw_data.groupby(['route', 'airline', 'aircraft_type']).agg(
    tow_count=('tow', 'count'),    # Count of takeoff weight records
    tow_mean=('tow', 'mean'),      # Mean takeoff weight
    tow_std=('tow', 'std')         # Standard deviation of takeoff weight
).reset_index()

# %%
# Calculate the percentage of variability in 'tow' by dividing standard deviation by mean
agg_df['tow_perc'] = agg_df['tow_std'] / agg_df['tow_mean']

# %%
# Display the aggregated DataFrame to verify calculations
agg_df

# %%
# Load predictions from two model versions for comparison, indexed by flight ID
pred_v9 = pd.read_csv('team_outspoken_engine_v0_796a128a-c833-453a-8653-5347905ae539.csv', index_col='flight_id')
pred_v11 = pd.read_csv('team_outspoken_engine_v1_796a128a-c833-453a-8653-5347905ae539.csv', index_col='flight_id')

# %%
# Display the first few rows of the version 11 predictions to verify data integrity
pred_v11.head()

# %%
# Merge the two prediction DataFrames on flight ID, adding suffixes to differentiate columns
merged_df = pred_v9.join(pred_v11, lsuffix='_v9', rsuffix='_v11')

# %%

# %%
# Calculate the average takeoff weight (tow) from both versions for each flight
merged_df['tow'] = (merged_df['tow_v9'] + merged_df['tow_v11']) / 2.0

# %%
# Save the merged predictions with the averaged 'tow' values to a new CSV file
merged_df['tow'].to_csv('./team_outspoken_engine_v2_796a128a-c833-453a-8653-5347905ae539.csv', index=True)

