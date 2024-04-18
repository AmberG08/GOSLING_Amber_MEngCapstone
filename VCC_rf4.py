import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from mapie.regression import MapieRegressor

from sklearn.preprocessing import PowerTransformer

np.random.seed(7)

#Boxcox
'''Load in file'''
df_main = pd.read_csv('CleanedData.csv')
df_main['Timestamp'] = pd.to_datetime(df_main['Timestamp'], format='%d/%m/%Y %H:%M')


'''Getting VCC datapoints'''
df_celldata = pd.DataFrame(df_main, columns=['Timestamp', 'VCC'])

no_nans = df_celldata[~df_celldata.isnull().any(axis=1)]
#print(no_nans)


actuals = df_celldata['VCC']

celldata=df_celldata['VCC'].to_numpy()
celldata = celldata.reshape((len(celldata),1))

boxcox = PowerTransformer(method='box-cox')
df_celldata['VCC'] = boxcox.fit_transform(celldata)

print('lambda = ', boxcox.lambdas_)

'''Separating data'''
time_sep=pd.to_datetime('2021-05-04 9:35:00')
df_log = df_celldata[df_celldata['Timestamp'] <= time_sep]
df_death = df_celldata[df_celldata['Timestamp'] >= time_sep]

df_log = df_log['VCC']
df_death = df_death['VCC']

nan_indices_log = np.isnan(df_log)
coefficients_log = np.polyfit(df_log.index[~nan_indices_log], df_log[~nan_indices_log], 3)
df_log= np.polyval(coefficients_log, df_log.index)
#print(df_log.shape)

nan_indices_death = np.isnan(df_death)
coefficients_death = np.polyfit(df_death.index[~nan_indices_death], df_death[~nan_indices_death], 1)
df_death = np.polyval(coefficients_death, df_death.index)
#print(df_death.shape)

'''Inverse box cox transform'''

""" Concatenation """
df_log = df_log[:-1] # cut out last datapoint to make them connect
df_main['VCC'] = boxcox.inverse_transform(pd.concat([pd.DataFrame(df_log), pd.DataFrame(df_death)], axis=0, ignore_index=True))

plt.rcParams.update({'font.size': 12})

plt.plot(df_main.index, df_main['VCC'], label='Polynomial fit', color='blue')
plt.scatter(actuals.index, actuals, label='Offline VCC', color='black')
plt.xlabel("Index")
plt.ylabel("VCC")

'''Cut out intial data''' ##Cut out as required
cut_date_time = pd.to_datetime('2021-04-30 14:45:00')
cut_index = df_main[df_main['Timestamp'] >= cut_date_time].index.min()

plt.axvline(x = cut_index, color = 'red', linestyle='--', label = 'Calibration stage')
plt.legend()
plt.show()

df_main = df_main[df_main['Timestamp'] >= cut_date_time]

""" Set features and target """
filtered_df=df_main.drop('Timestamp', axis=1)
y = filtered_df['VCC']
X = filtered_df.drop(['VCC'], axis=1)

'''Train-Test split'''
X_train, X_test, y_train, y_test = train_test_split(X, y,  train_size=0.8, test_size=0.2, random_state=10)

'''Scaling''' 
scaler = StandardScaler().fit(X_train) 
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

'''Fitting the model'''
model = RandomForestRegressor(n_estimators = 10, random_state = 0)

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

plt.rcParams.update({'font.size': 14})

plt.xlabel("Index")
plt.ylabel("VCC (×10^6 cells/mL)")
#plt.title("VCC Predictions against Index, Coverage = 95%")

'''Visualisation'''
plt.fill_between(sorted_comparison_df.index, sorted_comparison_df['Lower Bound'], sorted_comparison_df['Upper Bound'], label='Prediction Bands', color = 'orange', alpha = 0.8)

'''Scatter plot of index vs glucose'''
plt.scatter(sorted_comparison_df.index, sorted_comparison_df['Actual'], color = 'darkred', label = 'Actual Values', s=5)
plt.scatter(sorted_comparison_df.index, sorted_comparison_df['Predicted'], color = 'blue', label = 'Predictions', s=0.5)

ymin, ymax = 6, 20
plt.ylim(ymin, ymax)
# legend = plt.legend()
# legend.legend_handles[1]._sizes = [30]
# legend.legend_handles[2]._sizes = [30]
plt.show()

'''Feature Selection'''
features = ['Pred (X) 1 PV - Air Sparge', 'Pred (X) 2 PV - CO2 Sparge',	'Pred (X) 3 PV - O2 Sparge', 'Pred (X) 4 PV - N2 Sparge', 'Pred (X) 5 PV - Feed Flow', 'Resp (Y) 1 - pH', 'Resp (Y) 2 - DO', 'PV - F, Weight A', 'PV - F Weight B',	'PV - Temperature',	'CEDEX - GLC3B', 'Added Volume', 'Total Volume', 'Feed Added Since Last Sample']
f_i = list(zip(features,model.feature_importances_))
f_i.sort(key = lambda x : x[1])
plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
plt.tight_layout()
plt.show()

for feature, importance in f_i:
    print(f"{feature}: {importance:.5f}") 


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