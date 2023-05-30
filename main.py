import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import copy

from sklearn.model_selection import train_test_split
import xgboost as xgb 
import catboost
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from lightgbm import LGBMRegressor 

sample_submission = pd.read_csv('sample_submission.csv')
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

plt.figure(figsize=(15,10))
sns.heatmap(train.drop('id', axis=1).corr(), annot=True)

features_to_drop = ['id']
X = copy.deepcopy(train.drop(features_to_drop, axis=1))
y = X.pop('yield')
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

# CatBoost model
cat_model = catboost.CatBoostRegressor()
cat_model.fit(X_train, y_train, verbose=False)

# calculate MAE
cat_pred = cat_model.predict(X_valid)
cat_mae = metrics.mean_absolute_error(y_valid, cat_pred)


#LightGBM model
lgb_model = LGBMRegressor()
lgb_model.fit(X_train, y_train)

#calculate MAE
lgb_pred = lgb_model.predict(X_valid)
lgb_mae = metrics.mean_absolute_error(y_valid, lgb_pred)


# XGBoost model
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train, verbose=False)

# Predict on the test set and calculate MAE
xgb_pred = xgb_model.predict(X_valid)
xgb_mae = metrics.mean_absolute_error(y_valid, xgb_pred)


# Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

# calculate MAE
rf_pred = rf_model.predict(X_valid)
rf_mae = metrics.mean_absolute_error(y_valid, rf_pred)

print("CatBoost MAE: {:.2f}".format(cat_mae))
print("LightGBM MAE: {:.2f}".format(lgb_mae))
print("Random Forest MAE: {:.2f}".format(rf_mae))
print("XGBoost MAE: {:.2f}".format(xgb_mae))

# Fit LightGBM model
model = LGBMRegressor(num_leaves=10,
                      objective='regression_l1',
                      #learning_rate=0.01,
                      n_estimators=500,
                      max_depth=20,
                      min_child_samples=30,
                      )
model.fit(X_train, y_train)

# Predict on the test set and calculate MAE
y_pred = model.predict(X_valid)
mae = metrics.mean_absolute_error(y_valid, y_pred)
print('MAE: ', mae)

X_combined = pd.concat([X_train, X_valid], axis=0)
y_combined = pd.concat([y_train, y_valid], axis=0)
model.fit(X_combined,y_combined)

X_test = copy.deepcopy(test.drop(features_to_drop, axis=1))
y_pred_test = model.predict(X_test)
data = {'Id': test.id, 'yield': y_pred_test}
df = pd.DataFrame(data)

df.to_csv('submission_file.csv', index=False)