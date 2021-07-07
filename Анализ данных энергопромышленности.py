
# Анализ и визуализация данных

### General import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read input file
df = pd.read_csv('opsd_austria_daily.csv', index_col=0)
df.index = pd.to_datetime(df.index)
# Делаем срез данных для 2019 года
df1 = df[df.index.year == 2019]
# Task 1a: visualize distribution & time changes of input data
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
# Distribution of electricity consumption & wind production
ax[0,0].hist(df1['Electricity_consumption'], bins=50, color='palegreen', label='consumption') 
ax[0,0].set_xlabel('Consumption and wind power production, GWh')
ax[0,0].set_ylabel('Count')

ax2 = ax[0,0].twinx() 

ax2.hist(df1['Wind_production'], bins=50, color='lightblue', label='wind') 
ax2.set_ylabel('Count')
ax[0,0].legend(loc='best')
ax2.legend(loc='best')

# Distribution of solar power production
ax[0,1].hist(df1['Solar_production'], bins=50, color='darksalmon') 
ax[0,1].set_xlabel('Solar power production, GWh')
ax[0,1].set_ylabel('Count')

# Time series
ax[1,0].plot(df1['Electricity_consumption'], linewidth = 0.5, label='consumption') 
ax[1,0].plot(df1['Wind+Solar'], linewidth = 0.5, label='wind+solar production') 
ax[1,0].set_xlabel('Time')
ax[1,0].set_ylabel('Energy, GWh')
ax[1,0].legend(loc='best')

# Price distribution
ax[1,1].plot(df1['Price'], linewidth = 0.5) 
ax[1,1].set_xlabel('Time')
ax[1,1].set_ylabel('Price, Euro/1 kWh')
ax[1,1].legend(loc='best')
plt.show()


import warnings
warnings.filterwarnings('ignore')

''' defining a rolling average '''
df1['Electricity_consumption_RA'] = df1['Electricity_consumption'].rolling(15).mean()
df1['Wind_production_RA'] = df1['Wind_production'].rolling(15).mean()
df1['Solar_production_RA'] = df1['Solar_production'].rolling(15).mean()

# Plot
plt.figure(figsize=(12,8))
plt.subplot(2,2, 1)
ax = plt.plot(df1['Electricity_consumption_RA'])
plt.plot(df1['Electricity_consumption'], alpha=0.3)
plt.legend(['Electricity consumption RA', 'Electricity consumption'])
plt.title('Electricity consumption rolling avarage')
plt.xlabel('Date')
plt.ylabel('Value')
plt.subplot(2,2,2)
ax = plt.plot(df1['Wind_production_RA'])
plt.plot(df1['Wind_production'], alpha=0.3)
plt.legend(['Wind production RA', 'Wind_production'])
plt.title('Wind production rolling avarage')
plt.xlabel('Date')
plt.ylabel('Value')
plt.subplot(2,2,3)
ax = plt.plot(df1['Solar_production_RA'])
plt.plot(df1['Solar_production'], alpha=0.3)
plt.legend(['Solar production RA', 'Solar production'])
plt.title('Solar production rolling avarage')
plt.xlabel('Date')
plt.ylabel('Value');


# Использование Decision Tree + AdaBoost для прогнозирования временных рядов 
import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
# read input
df = pd.read_csv('opsd_austria_daily.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# data imputation
def clean(dataset, columns):
    dataset[columns].replace(0, np.nan, inplace=True)
    dataset[columns].fillna(method='backfill', inplace=True) # choose a scheme for fillna
# call a clean function here for Electricity_consumption, Wind_production columns
clean(df, 'Electricity_consumption')
clean(df, 'Wind_production')
# data imputation
def clean(dataset, columns):
    dataset[columns].replace(0, np.nan, inplace=True)
    dataset[columns].fillna(method='backfill', inplace=True) # choose a scheme for fillna
# call a clean function here for Electricity_consumption, Wind_production columns
clean(df, 'Electricity_consumption')
clean(df, 'Wind_production')
# ensemble learning for regression
def ensemble_training(df_train, df_test):
    label = 'Electricity_consumption'
    X_train = df_train.drop(label, axis=1)
    y_train = df_train.loc[:,label]
    X_test = df_test.drop(label, axis=1)
    y_test = df_test.loc[:,label]
    dtrab = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),
                              # check which value for max_depth gives better RMSE/MSE: 5, 10, 15, 20, 25
                              n_estimators=10,
                              # check which value for n_estimators gives better RMSE/MAE: 10, 20, 30, 40, 50, 100
                              random_state=1)
    dtrab.fit(X_train, y_train)
    y_predict=dtrab.predict(X_test)
    print('RMSE: %.6f' %(sqrt(mean_squared_error(y_test, y_predict))))
    print('MAE: %.6f' %(mean_absolute_error(y_test, y_predict)))
    df_sol=pd.DataFrame({'True': np.array(y_test),'Predicted': np.array(y_predict)})
    return dtrab, df_sol
features = ['Electricity_consumption','Wind_production','Month']
df_train = df.loc[df['Year']!=2019, features]
df_test = df.loc[df['Year']==2019, features]
def month_select(df, column):
    df_dummy = pd.get_dummies(df[column], prefix='M')
    df_new = pd.concat([df, df_dummy], axis=1)
    df_new = df_new.drop(column, axis=1)
    return df_new
df_train = month_select(df_train, 'Month')
df_test = month_select(df_test, 'Month')
df_sol = []
model, df_sol = ensemble_training(df_train, df_test)
df_sol = pd.concat([df_sol.reset_index(drop=True),
                  pd.Series(df.loc[df['Year']==2019,'Date']).reset_index(drop=True)], axis=1)
# visualization
fig,ax = plt.subplots(figsize=(10,4))
ax.plot_date(df_sol.loc[1:15,'Date'],
             df_sol.loc[1:15,'True'],
             marker='None',
             linestyle = '-',
             color='black', label='True')
ax.plot_date(df_sol.loc[1:15,'Date'],
             df_sol.loc[1:15,'Predicted'],
             marker='o',
             linestyle = '-',
             color='navy', markeredgecolor='navy', label='Predicted')
ax.set_xlabel('Time')
ax.set_ylabel('Electricity consumption')
ax.legend(loc='lower right')
plt.show()


# ### Использование GridSearchCV для подбора оптимальных гиперпараметров:
from sklearn.model_selection import GridSearchCV
estimator = AdaBoostRegressor()
param_grid = {'n_estimators': [5,10,15,20,25]}
cv=3
optimizer = GridSearchCV(estimator=estimator, param_grid=param_grid,cv=cv)
optimizer.fit(X_train, y_train)
print(optimizer.best_estimator_)
estimator = DecisionTreeRegressor()
param_grid = {'max_depth': [5, 10,20,30,40,50,100]}
optimizer = GridSearchCV(estimator=estimator, param_grid=param_grid,cv=cv)
optimizer.fit(X_train, y_train)
print(optimizer.best_estimator_)


# # Разработка бизнес-гипотез на основе анализа данных и оценка их экономического эффекта
# H0: Собственных ресурсов ветряной и солнечной энергии Австрии хватает, чтобы компенсировать спрос на электроэнергию страны.  
# H1: Собственных ресурсов ветряной и солнечной энергии Австрии не хватает, чтобы компенсировать спрос на электроэнергию страны, и придётся прибегать за к импорту природного газа
df['delta'] = df['Electricity_consumption'] - df['Wind+Solar']
print((df['delta']<0).sum())
df['Price'] = df['Price'].apply(lambda x: x*1000000)
df['Price_delta'] = df['Price'] * df['delta']
df.to_csv('opsd_austria_updated.csv')

