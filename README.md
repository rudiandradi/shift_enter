# Задания с виртуальных стажировок [SHIFT + ENTER by Changallenge](https://shift.changellenge.com/tasks)

## Рабочая среда: Jupyter Notebook
 *Описания заданий находятся в тетрадках Jupyter*

## Версии библиотек:
* tensorflow 2.5.0
* keras 2.5.0
* matplotlib 3.3.2
* pandas 1.1.3
* numpy 1.19.2
* sklearn 0.23.2
* seaborn 0.11.0
* statsmodels 0.12.0
* efficient_apriori 1.1.1


# 1. Поиск ассоциативных правил  
#### *Направление Data Analyst, компания P&G*
### Задачи:  
* Провести качественный анализ данных на полноту и оценить их качество;  
* Построить модель поиска ассоциативных правил (ARL) при помощи алгоритма apriori;  
* Изучить распределение по категориям данных; 
### Информация о датасете: 
Данные от P&G о истории покупок пользоваталей, содержат 309707 записей.
### Описание столбцов: 
* id чека  
* Дата и время совершения покупки  
* ИНН плательщика  
* Информация о пользователе (id)  
* Наименование товара  
* Наименование бренда  
* Теги, присвоенные товару  
* Цена товара  
* Количество единиц товара  
* Общая стоимость покупки  
* Сумма кэшбека
## Версии библиотек  
* pandas 1.1.3  
* numpy 1.19.2  
* efficient_apriori 1.1.1  
* matplotlib 3.3.2
## Выводы: 
1. Данные подгружаются с потерями, в 87% есть пустые значения.  
2. Согласно ассоциативному правилу, покупатели, приобретая товары личной гигены и прокладки склонны также добавлять в чек упаковки и пакеты.  
3. Топ 3 по популярности брендов P&G у покупателей это Always, Discreete и Gillette, где количество товаров Always в чеках превышает 175000. 

# 2. Анализ временных рядов и построение прогнозирующей модели ARIMA
#### Направление Data Analyst, компания P&G
## Задачи:
* Анализ трендов и сезонности данных путём работы с временными рядами.  
* Исследование данных на периодичность и построение графика автокорреляции.  
* Использование методов моделирования ARIMA для прогнозирования временных рядов.  
## Информация о датасете:
*Данные от компании P&G, содержат записи по месячным продажам шампуня*
## Описания столбцов: 
* Month - месяц, для которого релевантно наблюдение;  
* Количество проданных единиц товара; 
## Версии библиотек:  
* pandas 1.1.3  
* numpy 1.19.2  
* matplotlib 3.3.2  
* seaborn 0.11.0  
* statsmodels 0.12.0
## Выводы:  
1) Тренд, наблюдаемый в данных, на основании скользяшего среднего позитивный, увеличивается последние дней и уже достиг 550.     
2) Cезонность колеблется от ~-200 до ~280.  
3) Положительная автокорреляция наблюдается для лагов с первого по 11, отрицательная на всех остальных значениях.   
4) Ближе к концу третьего месяца автокорреляция выравнивается на нулевое значение, сохраняя позитивный тренд. 
