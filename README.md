# Задания с виртуальных стажировок [SHIFT + ENTER by Changallenge](https://shift.changellenge.com/tasks)

## Рабочая среда: Visual Studio
## Описания задания:
* [Поиск ассоциативных правил](https://github.com/rudiandradi/shift_enter#%D0%BF%D0%BE%D0%B8%D1%81%D0%BA-%D0%B0%D1%81%D1%81%D0%BE%D1%86%D0%B8%D0%B0%D1%82%D0%B8%D0%B2%D0%BD%D1%8B%D1%85-%D0%BF%D1%80%D0%B0%D0%B2%D0%B8%D0%BB)
* [Анализ временных рядов и построение прогнозирующей модели ARIMA](https://github.com/rudiandradi/shift_enter#%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7-%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D1%8B%D1%85-%D1%80%D1%8F%D0%B4%D0%BE%D0%B2-%D0%B8-%D0%BF%D0%BE%D1%81%D1%82%D1%80%D0%BE%D0%B5%D0%BD%D0%B8%D0%B5-%D0%BF%D1%80%D0%BE%D0%B3%D0%BD%D0%BE%D0%B7%D0%B8%D1%80%D1%83%D1%8E%D1%89%D0%B5%D0%B9-%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8-arima)
* [Анализ данных энергопромышленности](https://github.com/rudiandradi/shift_enter#%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7-%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85-%D1%8D%D0%BD%D0%B5%D1%80%D0%B3%D0%BE%D0%BF%D1%80%D0%BE%D0%BC%D1%8B%D1%88%D0%BB%D0%B5%D0%BD%D0%BD%D0%BE%D1%81%D1%82%D0%B8)
* [Обучение модели распознавать рукописные цифры](https://github.com/rudiandradi/shift_enter#%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5-%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8-%D1%80%D0%B0%D1%81%D0%BF%D0%BE%D0%B7%D0%BD%D0%B0%D0%B2%D0%B0%D1%82%D1%8C-%D1%80%D1%83%D0%BA%D0%BE%D0%BF%D0%B8%D1%81%D0%BD%D1%8B%D0%B5-%D1%86%D0%B8%D1%84%D1%80%D1%8B)

# Поиск ассоциативных правил  
#### *Направление Data Analyst, компания P&G*
#### Задачи:  
* Провести качественный анализ данных на полноту и оценить их качество;  
* Построить модель поиска ассоциативных правил (ARL) при помощи алгоритма apriori;  
* Изучить распределение по категориям данных; 
#### Информация о датасете: 
Данные от P&G о истории покупок пользоваталей, содержат 309707 записей.
#### Описание столбцов: 
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
#### Версии библиотек  
* pandas 1.1.3  
* numpy 1.19.2  
* efficient_apriori 1.1.1  
* matplotlib 3.3.2
#### Выводы: 
1. Данные подгружаются с потерями, в 87% есть пустые значения.  
2. Согласно ассоциативному правилу, покупатели, приобретая товары личной гигены и прокладки склонны также добавлять в чек упаковки и пакеты.  
3. Топ 3 по популярности брендов P&G у покупателей это Always, Discreete и Gillette, где количество товаров Always в чеках превышает 175000. 

# Анализ временных рядов и построение прогнозирующей модели ARIMA
#### *Направление Data Analyst, компания P&G*
#### Задачи:
* Анализ трендов и сезонности данных путём работы с временными рядами.  
* Исследование данных на периодичность и построение графика автокорреляции.  
* Использование методов моделирования ARIMA для прогнозирования временных рядов.  
#### Информация о датасете:
*Данные от компании P&G, содержат записи по месячным продажам шампуня*
#### Описания столбцов: 
* Month - месяц, для которого релевантно наблюдение;  
* Количество проданных единиц товара; 
#### Версии библиотек:  
* pandas 1.1.3  
* numpy 1.19.2  
* matplotlib 3.3.2  
* seaborn 0.11.0  
* statsmodels 0.12.0
#### Выводы:  
1) Тренд, наблюдаемый в данных, на основании скользяшего среднего позитивный, увеличивается последние дней и уже достиг 550.     
2) Cезонность колеблется от ~-200 до ~280.  
3) Положительная автокорреляция наблюдается для лагов с первого по 11, отрицательная на всех остальных значениях.   
4) Ближе к концу третьего месяца автокорреляция выравнивается на нулевое значение, сохраняя позитивный тренд. 

# Анализ данных энергопромышленности
#### *Направление Data Scientist, компания McKinsey*
#### Задачи: 
* Анализ сезонности данных, работая с временными рядами и используя Python;
* Визуализация данных с помощью построения графиков;  
* Написание Ensemble Machine Learning (ML) алгоритмов, в частности Decision Tree + AdaBoost, для прогнозирования временных рядов;  
* Использование методов градиентного спуска для подбора оптимальных гиперпараметров моделей;
* Выдвижение бизнес-гипотезы на основе анализа данных;
* Расчёт потенциального экономического эффекта бизнес гипотез.  
#### Информация о датасете
Открытый дата-сет, сгенерированный на основе информации [Open Power
System Data (OPSD)](https://data.open-power-system-data.org/time_series) для Австрии. Основными
источниками данных являются различные европейские операторы систем передачи (TSO). 
#### Описания столбцов   
Этот набор данных содержит значения ежедневного потребления энергии и ее
производства с помощью ветровых установок и солнечных панелей в течение
2015–2020 годов и даты:  
* Date: Дата в формате year-month-day.  
* Electricity_consumption: Потребление электричества в ГВт∙ч.  
* Wind_production: Производство ветровой энергии в ГВт∙ч.  
* Solar_production: Производство солнечной энергии в ГВт∙ч.  
* Price: Спотовая цена на электроэнергию для Австрии в евро за 1 кВт∙ч.  
* Wind+Solar: Сумма ветровой и солнечной энергии в ГВт∙ч.   
#### Версии библиотек:
* pandas 1.1.3
* numpy 1.19.2
* matplotlib 3.3.2
* sklearn 0.23.2 
#### Выводы:  
* Изменение производства ветровой энергии в течение 2019 года колеблется сильнее всего. Устойчиво, мньшие значения наблюдаются с июля по сентябрь, наибольшие же в весенние месяцы.    
* Изменение производства солнечной энергии в течение 2019 года возрастает с приближением к летним месяцам, с пиком июле (8 Г.в./час), когда дневное время максимально продолжительное.  В зимние месяцы находится практически на нуле.  
* Потребление электричества в течение 2019 года больше в зимние месяцы - самые высокие значения линии тренда приходятся на декабрь, январь и февраль. Однако даже в летние месяцы линия тренда возрастает (наверное, когда слишком жарко люди включают сплит-системы, или типа того).  
* Потребление энергии превышает её производство для всех полученных временных данных.

# Обучение модели распознавать рукописные цифры
#### Информация о датасете:
Объёмная база данных образцов рукописного написания цифр. База данных является стандартом, предложенным Национальным институтом стандартов и технологий США с целью калибрации и сопоставления методов распознавания изображений с помощью машинного обучения в первую очередь на основе нейронных сетей. 
#### Задачи:
* Чтение и преобразование данных  
* Настройка и проверка модели  
* Обучение на отложенных данных  
* Оценка точности
#### Версии библиотек:
* tensorflow 2.5.0
* keras 2.5.0
* matplotlib 3.3.2
