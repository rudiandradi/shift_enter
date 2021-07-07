
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Построение графика скользящего среднего и FOD
df = pd.read_csv(r'shampoo_sales.csv')
sales = df[['Sales']]
sales.plot(figsize=(12,10), linewidth=5, fontsize=20)
plt.xlabel('Month', fontsize=20),
plt.ylabel('Sales per month', fontsize=20)
plt.show()
import seaborn as sns
sns.set(
    font_scale=1.2,
    style='whitegrid',
    rc={'figure.figsize':(20,12)})
df['Rolling_mean'] = df['Sales'].rolling(6).mean()
df['FOD'] = df['Sales'].diff()
plt.subplot(2,2,1)
plt.plot(df['Rolling_mean'])
plt.xlabel('Month')
plt.ylabel('Rolling mean')
plt.title('Скользящее среднее продаж с окном 6')
plt.subplot(2,2,2)
plt.plot(df['FOD'])
plt.xlabel('Month')
plt.ylabel('First-order difference')
plt.title('Разность первого порядка продаж')
plt.show();

# Исследование периодичности и построение графика автокорреляции
pd.plotting.autocorrelation_plot(df['Sales'])
plt.show()


# # Построение модели скользящего прогноза ARIMA
import warnings
warnings.filterwarnings('ignore')
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
size = int(len(sales) * 0.6)
train_set, test_set = sales.values[0:size], sales.values[size:len(sales)]
history = [x for x in train_set]
predictions = list()
for i in range(len(test_set)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    pred = output[0]
    predictions.append(pred)
    obs = test_set[i]
    history.append(obs)
    print(f'predicted={round(pred[0],1)}, expected={obs}')
# plot
plt.plot(test_set)
plt.plot(predictions, color='red')
plt.legend(['Реальные данные','Предсказанные значения'])
plt.title('Предсказания модели')
plt.show()

