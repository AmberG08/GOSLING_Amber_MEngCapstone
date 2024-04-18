import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

np.random.seed(7)

'''Load in file'''
df_main = pd.read_csv('CleanedData.csv')

'''Cut out data from bioreactor calibration stage'''
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
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


'''Lasso regression'''
reg = Lasso(alpha=1) #arbitrary lambda of 1
reg.fit(X_train, y_train)

'''Prediction'''
pred_train = reg.predict(X_train)
mse_train = mean_squared_error(y_train, pred_train)

'''Testing'''
pred = reg.predict(X_test)
mse_test =mean_squared_error(y_test, pred)

'''lasso coefficients as a function of alpha'''
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

'''Lasso with 10 fold cross-validation to find best value of lambda'''
model = LassoCV(cv=10, random_state=0, max_iter=10000) #cv=10 means k=10.

model.fit(X_train, y_train) #Fitting model with Lasso CV
lasso_best = Lasso(alpha=model.alpha_) #Refit model with best value of lambda
lasso_best.fit(X_train, y_train)

print('\nBest value of lambda from 10-fold CV =', round(model.alpha_,5)) 

'''Getting coefficients for each feature'''
equation = [(round(coef, 3), feature) for coef, feature in zip(lasso_best.coef_, X)]
print(equation)

'''Model Evaluation'''
Rsq_train = lasso_best.score(X_train, y_train)
Rsq_test = lasso_best.score(X_test, y_test)
print('\nTrain R squared ', round(Rsq_train,5))
print('Test R squared', round(Rsq_test,5))

mse_train=mean_squared_error(y_train, lasso_best.predict(X_train))
mse_test=mean_squared_error(y_test, lasso_best.predict(X_test))
print('Train MSE ', round(mse_train,5))
print('Test MSE ', round(mse_test,5))

'''Visualising lowest MSE over 10-fold cross-validation'''
plt.semilogx(model.alphas_, model.mse_path_, ":") 
                                                  
plt.plot(
    model.alphas_ ,
    model.mse_path_.mean(axis=-1), 
    "k", #colour = black
    label="Average across the folds",
    linewidth=2,
)

plt.axvline( #vertical line where average MSE at at a minimum
    model.alpha_, linestyle="--", color="k", label="alpha: CV estimate"
)

plt.legend()
plt.xlabel("alphas")
plt.ylabel("Mean square error")
plt.title("Mean square error on each fold")
plt.axis("tight")

plt.show()

predictions = lasso_best.predict(X_test)
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

'''Conformal Prediction with MAPIE'''
from mapie.regression import MapieRegressor

mapie_regressor = MapieRegressor(estimator=lasso_best, method='plus', cv=10)

mapie_regressor = mapie_regressor.fit(X_train, y_train)
y_pred, y_pis = mapie_regressor.predict(X_test, alpha=[0.05]) #95% confidence level

comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Lower Bound': y_pis[:, 0, 0], 'Upper Bound': y_pis[:, 1, 0]})
sorted_comparison_df = comparison_df.sort_index()

"""Plotting predictions and actual values with conformal prediction bands"""
plt.rcParams.update({'font.size': 14})

plt.xlabel("Index")
plt.ylabel("Glucose (g/L)")

plt.fill_between(sorted_comparison_df.index, sorted_comparison_df['Lower Bound'], sorted_comparison_df['Upper Bound'], label='Prediction Bands', color = 'orange', alpha = 0.8)
plt.scatter(sorted_comparison_df.index, sorted_comparison_df['Actual'], color = 'darkred', label = 'Actual Values', marker="o", s=5)
plt.scatter(sorted_comparison_df.index, sorted_comparison_df['Predicted'], color = 'blue', label = 'Predictions', s=0.5)

ymin, ymax = 2, 6
plt.ylim(ymin, ymax)
# legend = plt.legend()
# legend.legend_handles[1]._sizes = [30]
# legend.legend_handles[2]._sizes = [30]
plt.show()

'''Calculating metrics to evaluate conformal prediction (Upperbound - Lowerbound)/actuals'''
cp_metric = ((sorted_comparison_df['Upper Bound'] - sorted_comparison_df['Lower Bound']) / sorted_comparison_df['Actual'])*100

cp_metric_min = np.min(cp_metric)
cp_metric_max = np.max(cp_metric)
cp_metric_mean = np.mean(cp_metric)

print("\nCP metrics")
print("Min:", round(cp_metric_min,3))
print("Max:", round(cp_metric_max,3))
print("Mean:", round(cp_metric_mean,3))