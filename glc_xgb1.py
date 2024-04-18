import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,  r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV

from mapie.regression import MapieRegressor

from xgboost import XGBRegressor

np.random.seed(7)

'''Load in file'''
df_main = pd.read_csv('CleanedData.csv')

'''Cut out first ~5000 data points'''
df_main['Timestamp'] = pd.to_datetime(df_main['Timestamp'], format='%d/%m/%Y %H:%M')
cut_date_time = pd.to_datetime('2021-04-30 14:45:00')
filtered_df = df_main[df_main['Timestamp'] >= cut_date_time]

filtered_df=filtered_df.drop('Timestamp', axis=1)
filtered_df=filtered_df.drop('VCC', axis=1)

'''Set target variable'''
y = filtered_df['CEDEX - GLC3B'] #Target variable
X = filtered_df.drop(['CEDEX - GLC3B'], axis=1)

'''Train-Test split'''
X_train, X_test, y_train, y_test = train_test_split(X, y,  train_size=0.8, test_size=0.2, random_state=10)

'''Scaling''' 
scaler = StandardScaler().fit(X_train) #Computes mean and st. dev
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

'''Fitting the model'''
model = XGBRegressor(n_estimators = 10, random_state = 0) #10 trees

# Fitting the Random Forest Regression model to the data
model.fit(X_train, y_train)

'''MAPIE'''
mapie_regressor = MapieRegressor(estimator=model, method='plus', cv=10)
mapie_regressor = mapie_regressor.fit(X_train, y_train)

y_pred, y_pis = mapie_regressor.predict(X_test, alpha=[0.05]) #95% confidence level

comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Lower Bound': y_pis[:, 0, 0], 'Upper Bound': y_pis[:, 1, 0]})
sorted_comparison_df = comparison_df.sort_index()

'''Model Evaluation'''
print('\nR squared training set', round(mapie_regressor.score(X_train, y_train), 5))
print('R squared test set', round(mapie_regressor.score(X_test, y_test), 5))

'''MSE'''
mse_train=mean_squared_error(y_train, mapie_regressor.predict(X_train))
mse_test=mean_squared_error(y_test, mapie_regressor.predict(X_test))
print('Train MSE ', round(mse_train, 5))
print('Test MSE ', round(mse_test, 5))

"""Plotting"""

plt.xlabel("Index")
plt.ylabel("Glucose")

'''Visualisation'''
plt.fill_between(sorted_comparison_df.index, sorted_comparison_df['Lower Bound'], sorted_comparison_df['Upper Bound'], label='Prediction Bands', color = 'orange', alpha = 0.8)

'''Scatter plot of index vs glucose'''
plt.scatter(sorted_comparison_df.index, sorted_comparison_df['Actual'], color = 'darkred', label = 'Actual Values', s=5)
plt.scatter(sorted_comparison_df.index, sorted_comparison_df['Predicted'], color = 'blue', label = 'Predictions', s=0.5)

ymin, ymax = 3, 5.5
plt.ylim(ymin, ymax)
# legend = plt.legend()
# legend.legend_handles[1]._sizes = [30]
# legend.legend_handles[2]._sizes = [30]
plt.show()

'''Feature Selection'''
features = ['Pred (X) 1 PV - Air Sparge',	'Pred (X) 2 PV - CO2 Sparge',	'Pred (X) 3 PV - O2 Sparge',	'Pred (X) 4 PV - N2 Sparge',	'Pred (X) 5 PV - Feed Flow',	'Resp (Y) 1 - pH',	'Resp (Y) 2 - DO',	'PV - F, Weight A',	'PV - F Weight B',	'PV - Temperature',	'Added Volume',	'Total Volume',	'Feed Added Since Last Sample']
f_i = list(zip(features,model.feature_importances_))
f_i.sort(key = lambda x : x[1])

for feature, importance in f_i:
    print(f"{feature}: {importance:.5f}")  
    
plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
plt.tight_layout()
plt.show()

'''CP Metrics''' 
# Calculating (Upperbound - Lowerbound)/actuals
cp_metric = ((sorted_comparison_df['Upper Bound'] - sorted_comparison_df['Lower Bound']) / sorted_comparison_df['Actual'])*100

# Outputting minimum, maximum, and mean of the calculation
cp_metric_min = np.min(cp_metric)
cp_metric_max = np.max(cp_metric)
cp_metric_mean = np.mean(cp_metric)

print("Minimum:", round(cp_metric_min,3))
print("Maximum:", round(cp_metric_max,3))
print("Mean:", round(cp_metric_mean,3))




