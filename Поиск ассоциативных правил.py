import pandas as pd
import numpy as np
df = pd.read_csv(r'Case_Data.csv', sep=',')
df.head(4)

# Поиск пропущенных значений
nans = df.isna().sum(axis=0)
print('Количество пропущенных значений в каждом столбце:\n', df.isna().sum(), '\n\n')

shape = len(df)
percent_list = []
for i in nans:
    ans = i / shape
    ans = round(ans, 5)
    procent = ans * 100
    answer = round(procent, 3)
    percent_list.append(answer)
print('Доля пропущенных значений в каждом столбце:')
total = list(zip(df.columns, percent_list))
for i in total:
    print(f'{i[0]} - {i[1]}% пропущенных значений')
print('\n\nКоличество строк, содержащих пропущенные значения:\n', df.isna().any(axis=1).sum(), '\n\n')
print('Доля строк, содержащих пропущенные значенния:\n', round((df.isna().any(axis=1).sum() / df.shape[0]) * 100, 4), '%')
    

# Поиск ассоциативных правил
cols = ['Бренд', 'Теги', 'Цена', 'Кол-во', 'Сумма', 'Кэшбэк']
df[cols] = df[cols].fillna(df.mode().iloc[0])
df2 = df.groupby(df.columns[0])['Теги'].unique()
print(df2.head(),'\n\n')
df3 = list(map(tuple,df2))
print(df3[0:2])


# Apriori
from efficient_apriori import apriori
association_rules = apriori(df3, min_support=0.007,
                            min_confidence=0.0045) 
print(association_rules[1][0:5],'\n\n') 
print(association_rules[0], '\n\n')
print(len(association_rules))

for item in association_rules:
    pair = item[1]
    print(pair, '\n\n\n') # lift, support, confidence для одного правила

# Поиск товаров P&G в чеках покупателей

df['Бренд'] = df['Бренд'].str.lower()
brand_list = ['ariel', 'tide', 'миф', 'lenor', 'always', 'tampax', 'naturella','discreet', 'braun', 'gillette',
             'venus', 'head&shoulders', 'pantene', 'herbal', 'aussie', 'fairy', 'mr. proper', 'oral-b',
             'blend-a-med', 'clearblue', 'old spice', 'safeguard', 'pampers', 'bear fruits']

def count_brand(dataset, brand_name, parser): 
    found_rows = df['Бренд'].str.contains(parser).sum() 
    print(brand_name + ":", "%0.0f" % df['Бренд'].str.contains(parser).sum()) 
    dct[i] = found_rows
    return found_rows

dct = {}
for i in brand_list:
    count_brand(df, i, i)
res = pd.Series(dct)
res.sort_values(ascending=False).head(3)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(20,12))
plt.barh(res.index, res)
plt.title('Топ товаров P&G в чеках покупателей')
plt.ylabel('Товар')
plt.xlabel('Количество товаров')


# # Выводы: 
#   
# 1. Данные подгружаются с потерями, в 87% есть пустые значения.  
# 2. Согласно ассоциативному правилу, покупатели, приобретая товары личной гигены и прокладки склонны также добавлять в чек упаковки и пакеты.  
# 3. Топ 3 по популярности брендов P&G у покупателей это Always, Discreete и Gillette, где количество товаров Always в чеках превышает 175000. 




