import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PowerTransformer

np.random.seed(7)

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

'''Box Cox transformation'''
boxcox = PowerTransformer(method='box-cox')
df_celldata['VCC'] = boxcox.fit_transform(celldata)

print('lambda = ', boxcox.lambdas_)

'''Separating data into logarithmic and cell death phases'''
time_sep=pd.to_datetime('2021-05-04 9:35:00')
df_log = df_celldata[df_celldata['Timestamp'] <= time_sep]
df_death = df_celldata[df_celldata['Timestamp'] >= time_sep]

df_log = df_log['VCC']
df_death = df_death['VCC']

nan_indices_log = np.isnan(df_log)
coefficients_log = np.polyfit(df_log.index[~nan_indices_log], df_log[~nan_indices_log], 3) #Third order polynomal on non-nan datapoints
df_log= np.polyval(coefficients_log, df_log.index)
#print(df_log.shape)

nan_indices_death = np.isnan(df_death)
coefficients_death = np.polyfit(df_death.index[~nan_indices_death], df_death[~nan_indices_death], 1) #First order polynomial
df_death = np.polyval(coefficients_death, df_death.index)
#print(df_death.shape)

'''Inverse box cox transform'''
""" Concatenation of cell growth phases"""
df_log = df_log[:-1] # cut out last datapoint to allow concatenation 
df_main['VCC'] = boxcox.inverse_transform(pd.concat([pd.DataFrame(df_log), pd.DataFrame(df_death)], axis=0, ignore_index=True))

plt.rcParams.update({'font.size': 12})

plt.plot(df_main.index, df_main['VCC'], label='Polynomial fit', color='blue')
plt.scatter(actuals.index, actuals, label='Offline VCC', color='darkred')
plt.xlabel("Index")
plt.ylabel("VCC (×10^6 cells/mL)")

'''Cut out intial data'''
cut_date_time = pd.to_datetime('2021-04-30 14:45:00')
cut_index = df_main[df_main['Timestamp'] >= cut_date_time].index.min()

plt.axvline(x = cut_index, color = 'red', linestyle='--', label = 'Calibration stage')
plt.legend()
plt.show()

df_main = df_main[df_main['Timestamp'] >= cut_date_time]

# plt.plot(df_main.index, df_main['VCC'], label='Boxcox polyval', color='black')
# #plt.scatter(df_celldata.index, df_celldata['VCC'], label='Boxcox transformed Data', color='black' )
# plt.xlabel("Index")
# plt.ylabel("Box cox VCC")
# plt.legend()
# plt.show()

""" Set features and target """
filtered_df=df_main.drop('Timestamp', axis=1)
y = filtered_df['VCC']
X = filtered_df.drop(['VCC'], axis=1)

'''Train-Test split'''
X_train, X_test, y_train, y_test = train_test_split(X, y,  train_size=0.8, test_size=0.2, random_state=10)

'''Scaling''' 
scaler = StandardScaler().fit(X_train) #Computes mean and st. dev
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

'''Lasso regression'''
reg = Lasso(alpha=1) #larger alpha -> more aggressive penalisation
reg.fit(X_train, y_train)

'''Prediction'''
#pred_train = reg.predict(X_train)
#mse_train = mean_squared_error(y_train, pred_train) #comparing y_training data to y_predicted data
#print('Training MSE ', round(mse_train, 2))

'''Testing'''
#pred = reg.predict(X_test)
#mse_test =mean_squared_error(y_test, pred)
#print('Test MSE ', round(mse_test, 2))


'''lasso coefficients as a function of lambda'''
alphas = np.linspace(0.001,10,100)
lasso = Lasso(max_iter=10000)
coefs = []

for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs.append(lasso.coef_)

ax = plt.gca() 
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('lambda')
plt.ylabel('Standardized Coefficients')
plt.title('Lasso coefficients as a function of lambda')
plt.show()

'''Lasso with 5 fold cross-validation'''
model = LassoCV(cv=10, random_state=0, max_iter=10000) 

model.fit(X_train, y_train) #Fitting model with Lasso CV

lasso_best = Lasso(alpha=model.alpha_) #Refit model with best value of lambda
lasso_best.fit(X_train, y_train)

print('Best value of apha from CV =', round(model.alpha_,5)) #best value of lambda
equation = [(round(coef, 3), feature) for coef, feature in zip(lasso_best.coef_, X)]
print(equation)
#print(list(zip(lasso_best.coef_, X)))

'''Model Evaluation'''
Rsq_train = lasso_best.score(X_train, y_train)
Rsq_test = lasso_best.score(X_test, y_test)
print('\nTrain R squared ', round(Rsq_train,5))
print('Test R squared', round(Rsq_test,5))

plt.semilogx(model.alphas_, model.mse_path_, ":")
plt.plot(
    model.alphas_ ,
    model.mse_path_.mean(axis=-1), #axis=-1, calculated along last axis of array.
    "k", #colour = black
    label="Average across the folds",
    linewidth=2,
)

plt.axvline( #vertical line
    model.alpha_, linestyle="--", color="k", label="alpha: CV estimate"
)

plt.legend()
plt.xlabel("lambdas")
plt.ylabel("Mean square error")
plt.title("Mean square error on each fold")
plt.axis("tight")

#ymin, ymax = 0.2, 0.3
#plt.ylim(ymin, ymax)
plt.show()


predictions = lasso_best.predict(X_test)

comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

'''MAPIE'''
from mapie.regression import MapieRegressor

mapie_regressor = MapieRegressor(estimator=lasso_best, method='plus', cv=5)

mapie_regressor = mapie_regressor.fit(X_train, y_train)
y_pred, y_pis = mapie_regressor.predict(X_test, alpha=[0.05]) #95% confidence level

comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Lower Bound': y_pis[:, 0, 0], 'Upper Bound': y_pis[:, 1, 0]})
sorted_comparison_df = comparison_df.sort_index()
#print(sorted_comparison_df.to_string())

"""Plotting"""

plt.xlabel("Index")
plt.ylabel("VCC (×10^6 cells/mL)")

plt.fill_between(sorted_comparison_df.index, sorted_comparison_df['Lower Bound'], sorted_comparison_df['Upper Bound'], label='Prediction Bands', color = 'orange', alpha = 0.8)
plt.scatter(sorted_comparison_df.index, sorted_comparison_df['Actual'], color = 'darkred', label = 'Actual Values', s=5)
plt.scatter(sorted_comparison_df.index, sorted_comparison_df['Predicted'], color = 'blue', label = 'Predictions', s=0.5)

ymin, ymax = 6, 20
plt.ylim(ymin, ymax)
# legend = plt.legend()
# legend.legend_handles[1]._sizes = [30]
# legend.legend_handles[2]._sizes = [30]
# plt.show()
plt.show()


'''CP metrics'''
cp_metric = ((sorted_comparison_df['Upper Bound'] - sorted_comparison_df['Lower Bound']) / sorted_comparison_df['Actual'])*100

cp_metric_min = np.min(cp_metric)
cp_metric_max = np.max(cp_metric)
cp_metric_mean = np.mean(cp_metric)

print("\nCP metrics")
print("Min:", round(cp_metric_min,3))
print("Max:", round(cp_metric_max,3))
print("Mean:", round(cp_metric_mean,3))