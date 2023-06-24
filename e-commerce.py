#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[63]:


сustomers = pd.read_csv('olist_customers_dataset.csv', sep=',')


# In[64]:


сustomers.head()


# In[65]:


orders = pd.read_csv('olist_order_items_dataset.csv', sep=',')
orders.head()


# In[66]:


statuses = pd.read_csv('olist_orders_dataset.csv', sep=',')
statuses.head()


# # Сколько у нас пользователей, которые совершили покупку только один раз? (7 баллов) 

# In[68]:


purchases = сustomers.merge(statuses, on = 'customer_id')
purchases.head()
# Объединим датафреймы сustomers и statuses, чтобы посмотреть количество совершенных покупок пользователями


# In[69]:


purchases.shape
# Посмотрим размер данных


# In[70]:


purchases.dtypes
# Тип данных


# In[71]:


purchases.isna().sum()
# Проверим пропущенные значения 


# In[72]:


purchases.query('order_status == "delivered"').isna().sum()
# Проверим пропущенные значения у заказов со статусом доставлен
# У 14-ти заказов нет времени подтверждения оплаты, у 2-ух - времени передеачи в лог-ую службу, у 8-ми - времени доставки.

Посмотрим, сколько уникальных пользователей совершили только одну покупку. 

Совершенной покупкой считается доставленный и оплаченный заказ,
поэтому применим фильтры по статусу 'доставлен' и наличию значения в колонке время подтверждения оплаты.
Но, так как в датафрейме есть заказы со статусом доставлен, но без времени доставки или времени передачи в логистическую службу,
отфильтруем также по наличию значений в этих колонках (заказы с пустыми значениями в этих колонках не считаем совершенной покупкой)
# In[74]:


purchases.query('order_status == "delivered" and order_approved_at.notnull() and order_delivered_customer_date.notnull() and order_delivered_carrier_date.notnull()') .groupby('customer_unique_id') .agg({'order_id':'count'}) .query('order_id == 1') .order_id.sum()


# # 90536 уникальных пользователей совершили только одну покупку.

# In[ ]:





# # 2. Сколько заказов в месяц в среднем не доставляется по разным причинам (вывести детализацию по причинам)? (10 баллов)

# In[75]:


orders_and_statuses = statuses.query("order_status != 'delivered' and order_delivered_customer_date.isna()")
orders_and_statuses
#Делаем фильтрацию по статусу заказа - не равно доставлен, и отсутствию значения в колонке время доставки.


# In[76]:


orders_and_statuses.order_id.nunique()
# Прверим - нет ли повторяющихся значений order_id


# In[78]:


status = orders_and_statuses.groupby('order_status', as_index = False) .agg({'order_id':'count'})
status['percent'] = (status.order_id/status.order_id.sum()*100).round(2)
status
# Посмотрим, сколько у нас всего недоставленных заказов по количеству и по доле от общей суммы в % с разбивкой по статусу.


# In[79]:


orders_and_statuses.dtypes
# Проверим тип данных в колонках со временем


# In[83]:


orders_and_statuses['order_estimated_delivery_date'] = pd.to_datetime(orders_and_statuses['order_estimated_delivery_date'])
orders_and_statuses['order_purchase_timestamp'] = pd.to_datetime(orders_and_statuses['order_purchase_timestamp'])
orders_and_statuses['order_approved_at'] = pd.to_datetime(orders_and_statuses['order_approved_at'])
orders_and_statuses['order_delivered_carrier_date'] = pd.to_datetime(orders_and_statuses['order_delivered_carrier_date'])
orders_and_statuses['order_delivered_customer_date'] = pd.to_datetime(orders_and_statuses['order_delivered_customer_date'])
# Приведем данные в колонках со временем к формату datetime


# In[84]:


orders_and_statuses['year'] = orders_and_statuses['order_estimated_delivery_date'].dt.year
orders_and_statuses['month'] = orders_and_statuses['order_estimated_delivery_date'].dt.month
# Создадим новые колонки с данными "год" и "месяц" для подсчета количества месяцев в датасете


# In[85]:


# Посмотрим, сколько всего месяцев в наших данных
# Для этого возьмем данные из исходного датафейма, чтобы посмотреть кол-во месяцев, где есть значения 
# в колонке с обещанной датой доставки
month = statuses
month['order_estimated_delivery_date'] = pd.to_datetime(month['order_estimated_delivery_date'])
month['year'] = month['order_estimated_delivery_date'].dt.year
month['month'] = month['order_estimated_delivery_date'].dt.month
month.groupby(['year', 'month'], as_index = False) .agg({'order_id':'count'}) .shape
# 27 месяцев


# In[86]:


# Посчитаес для каждого отдельного статуса количество недоставленных заказов
# И посчитаем их среднее кол-во  в месяц:
status_mean = orders_and_statuses.groupby('order_status', as_index = False).agg({'order_id':'count'})
status_mean['mean'] = status_mean.order_id/27
status_mean


# In[87]:


grafic = orders_and_statuses.groupby(['year', 'month','order_status'], as_index = False) .agg({'order_id':'count'})


# In[88]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(data=grafic, x="order_status", y="order_id")

# Посмотрим на графике распределение надоставленных заказов в разбивке по статусам
# Большинство недоставленных заказов (~37% от общего числа недоставленных, 41 шт в среднем в месяц) со статусом 'shipped'.


# In[89]:


orders_and_statuses.query('order_status == "shipped"').isna().sum()
#Посмотрим пропущенные значения для заказов со статусом "shipped".

1107
Столько же, сколько и всего таких заказов. Все заказы были переданы в логистическую службу 
и в конечном итоге не были доставлены.
# In[ ]:





# # 3. По каждому товару определить, в какой день недели товар чаще всего покупается. (7 баллов)

# In[91]:


products = orders.merge(statuses, on = 'order_id')
products.head()
# Объединим датафреймы orders и statuses


# In[92]:


products.product_id.nunique()
# Посмотрим, сколько всего уникальных тотваров


# In[93]:


products['order_purchase_timestamp'] = pd.to_datetime(products['order_purchase_timestamp'])
# Сделаем колонку со временем создания заказа типа datetime


# In[94]:


products['weekday'] = products['order_purchase_timestamp'].dt.day_name()
# Создадим новую колонку со значениями день недели


# In[95]:


products.head(5)


# In[96]:


df_weekday = products.query('order_status == "delivered" and order_approved_at.notnull() and order_delivered_customer_date.notnull() and order_delivered_carrier_date.notnull()') .groupby(['product_id', 'weekday'], as_index = False) .agg({'order_item_id':'count'})
# Создадим новый датафрейм, чтобы посмотреть для каждого товара и дня недели - сколько раз он покупался.
# Применим фильтр из первого задания, так как покупкой мы считаем доставленный, оплаченный товар с наличием значений 
# во всех колонках со временем


# In[97]:


df_weekday


# In[98]:


df_weekday.query('product_id == "fffdb2d0ec8d6a61f0a0a0db3f25b441"')
# Посмотрим на примере одного товара, сколько раз он покупался в разные дни недели.
# Во вторник данный товар покупался максимальное кол-во раз.


# In[99]:


df_weekday.order_item_id.max()
# Посмотрим максимальное количество покупок одного товара.
# Максимальное количество покупок какого-то товара - 93


# In[100]:


df_weekday.query('order_item_id == "93"')
# 93 раза его купили в среду


# In[101]:


# Создадим дф с днем недели для кажого товара, в который его купили максимальное кол-во раз.
max_purs_product = df_weekday.groupby('product_id', as_index = False)['weekday'] .agg({'order_item_id':'max'})
max_purs_product


# # 4. Сколько у каждого из пользователей в среднем покупок в неделю (по месяцам)? Не стоит забывать, что внутри месяца может быть не целое количество недель. Например, в ноябре 2021 года 4,28 недели. И внутри метрики это нужно учесть. (8 баллов) 

# In[102]:


customers_and_statuses = сustomers.merge(statuses, on = 'customer_id')
# Объединим cuctomers and statuses


# In[103]:


customers_and_statuses


# In[104]:


duplicateRows = customers_and_statuses[customers_and_statuses.duplicated ()]
duplicateRows
# Проверим на наличие дубликатов


# In[105]:


customers_and_statuses.isna().sum()
# И на наличие пропущенных значений 


# In[106]:


customers_and_statuses = customers_and_statuses.query('order_status == "delivered" and order_approved_at.notnull() and order_delivered_customer_date.notnull() and order_delivered_carrier_date.notnull()')
# Применим уже знакомые фильтры, чтобы оставить в датафрейме только купленные товары


# In[107]:


customers_and_statuses.drop(customers_and_statuses.columns[[0, 2, 3, 4, 6, 8, 9, 10, 11, 12]], axis = 1, inplace=True)
# Удалим ненужные колонки 


# In[108]:


customers_and_statuses['order_purchase_timestamp'] = pd.to_datetime(customers_and_statuses['order_purchase_timestamp'])
# День покупки будем считать день создания заказа, приведем колонку с этими данными к типу datetime


# In[109]:


customers_and_statuses['month'] = customers_and_statuses['order_purchase_timestamp'].dt.month
customers_and_statuses['year'] = customers_and_statuses['order_purchase_timestamp'].dt.year
# Создадим новые колонки "месяц" и "год"


# In[110]:


customers_and_statuses['days'] = customers_and_statuses['order_purchase_timestamp'].dt.days_in_month
#Создадим новую колонку с количеством дней в заданном месяце, чтобы далее посчитать кол-во недель в этом месяце


# In[120]:


customers_and_statuses['weeks'] = (customers_and_statuses['days']/7).round(2)
# Создадим новую колонку с количеством недель


# In[121]:


customers_and_statuses


# In[134]:


df_purshape = customers_and_statuses.groupby(['customer_unique_id','year', 'month', 'weeks'], as_index = False) .agg({'order_id': 'count'})
# Сделаем новый дф с группировкой по уник-му айди покупателя, году-месяцу-кол-ву недель в месяце 
# и посчитаем для каждого количество покупок


# In[135]:


df_purshape['mean'] = (df_purshape['order_id']/df_purshape['weeks']).round(2)
# Создадим новую колонку, где посчитаем среднее кол-во заказов для каждого пользователя в каждом месяце


# In[136]:


df_purshape


# In[138]:


df_purshape.query('customer_unique_id == "c8460e4251689ba205045f3ea17884a1"')
# Посмотрим среднее для какого-то клиента
# в августе 2018 он сделал 4 заказа, в среднем в неделю для этого месяца - это 0.9 заказов.


# In[139]:



sns.displot(data=df_purshape, x="mean", kind="kde", height=8, aspect=2)
# Посмотрим на графике распределение у пользоватедей количества средних покупок в неделю по месяцам
# Большинство пользователей в среднем совершали не болле 1 покупки в неделю.


# In[130]:


df_purshape.query('mean > 1') .customer_unique_id.nunique()
#Всего 1 уник-ый пользователm совершил в среднем в неделю больше 1 покупки.


# # 5. Используя pandas, проведи когортный анализ пользователей. В период с января по декабрь выяви когорту с самым высоким retention на 3й месяц. Описание подхода можно найти тут. (15 баллов)

# In[143]:


df_cust_stat = сustomers.merge(statuses, on = 'customer_id')
# Объединим сustomers и statuses


# In[144]:


df_cust_stat = df_cust_stat.query('order_status == "delivered" and order_approved_at.notnull() and order_delivered_customer_date.notnull() and order_delivered_carrier_date.notnull()')
# Применим всё те же фильтры, чтобы оставить в датафрейме только купленные товары


# In[145]:


df_cust_stat.drop(df_cust_stat.columns[[0, 2,3, 4, 6, 8, 9, 10, 11]], axis = 1, inplace=True)
# Удалим ненужные колонки


# In[146]:


df_cust_stat['order_purchase_timestamp'] = pd.to_datetime(df_cust_stat['order_purchase_timestamp'])
# приведем колонку order_purchase_timestamp к типу datetime


# In[147]:


df_cust_stat['year_month'] = df_cust_stat['order_purchase_timestamp'].dt.to_period("M").astype(int)

# Создадим новую колонку год-месяц покупки и приведем к типу int для дальнейщего расчета дельты


# In[148]:


df_cust_stat["first_purсh"] = df_cust_stat.groupby('customer_unique_id')['order_purchase_timestamp'].transform('min')
# Cоздадим новую колонку с датой первой покупки 


# In[149]:


df_cust_stat['month_first_purсh'] = pd.to_datetime(df_cust_stat["first_purсh"]).dt.month
# Cоздадим новую колонку с номером месяца первой покупки пользователя


# In[150]:


df_cust_stat['year_month_first_purсh'] = df_cust_stat['first_purсh'].dt.to_period("M").astype(int)

# Cоздадим новую колонку с периодом (год-месяц) первой покупки пользователя и приведем к типу int для дальнейщего расчета дельты


# In[151]:


df_cust_stat = df_cust_stat.query(' "2017-01-01" <= first_purсh <= "2017-12-31"')
# Оставим для анализа данные по пользователем, у которых первая покупка была совершена не ранее января и не позднее декабря 2017


# In[152]:


df_cust_stat = df_cust_stat.query('order_purchase_timestamp <= "2018-02-28"')
# Также оставим данные о повторных покупках не позже февраля 2018 (третий месяц от декабря 2017)


# In[153]:


df_cust_stat.head()


# In[154]:


month_first = df_cust_stat.groupby(['month_first_purсh', 'year_month_first_purсh', 'year_month'], as_index = False) .agg({'customer_unique_id':'nunique'})
month_first.head()
# Сгруппируем данные по месяцу первой покупки, и по значениям для подсчета дельты
# и посчитаем сколько уникальных пользователей сделали заказ


# In[155]:


month_first['period'] = (month_first.year_month - month_first.year_month_first_purсh)
# Добавим колонку период, чтобы видеть на какой месяц от первой покупки, был сделан повторный заказ


# In[156]:


month_first


# In[157]:


periods = month_first.pivot(index='month_first_purсh', columns = 'period', values = 'customer_unique_id')
periods = periods.reset_index()
periods
# Сделаем сводную таблицу


# In[158]:


periods['ret'] = periods[2]/periods[0]

# Добавим колонку ret со значением retention (поделим кол-во ун-ых пользователей совершивших 
# повторную покупку на третий месяц на кол-во пол-ей, совершивших покупку впревые)


# In[159]:


periods


# In[575]:


periods.ret.max()

Маскимальное значение retention 3-го месяца - 0.005495878091431427, это значение соостветсвует девятому месяцу первой покупки, 
соответственно  когорта с самым высоким retention наблюдается в сентябре 2017 года.
# In[160]:


sns.lineplot(data=periods, x="month_first_purсh", y = 'ret')

# На графике можно посмотреть изменение retention по месяцам
# Самые низкие значения наблюдаются в  январе, апреле, октябре и декабре; самые высокие - марте, мае, сентябре и ноябре


# In[ ]:





# # 6. Часто для качественного анализа аудитории использую подходы, основанные на сегментации. Используя python, построй RFM-сегментацию пользователей, чтобы качественно оценить свою аудиторию. В кластеризации можешь выбрать следующие метрики: R - время от последней покупки пользователя до текущей даты, F - суммарное количество покупок у пользователя за всё время, M - сумма покупок за всё время. Подробно опиши, как ты создавал кластеры. Для каждого RFM-сегмента построй границы метрик recency, frequency и monetary для интерпретации этих кластеров. Пример такого описания: RFM-сегмент 132 (recency=1, frequency=3, monetary=2) имеет границы метрик recency от 130 до 500 дней, frequency от 2 до 5 заказов в неделю, monetary от 1780 до 3560 рублей в неделю. Описание подхода можно найти тут. (23 балла)

# In[7]:


df_6 = сustomers.merge(statuses, on = 'customer_id')
# Объединим все три датафрейма


# In[8]:


cohort = df_6.merge(orders, on = 'order_id')


# In[9]:


cohort = cohort.query('order_status == "delivered" and order_approved_at.notnull() and order_delivered_customer_date.notnull() and order_delivered_carrier_date.notnull()')
# Применим фильтры, чтобы оставить в датафрейме только купленные товары


# In[10]:


cohort.head()


# In[11]:


duplicateRows = cohort[cohort.duplicated ()]
duplicateRows
# Проверим на наличие дубликатов


# In[12]:


cohort.isna().sum()
# Проверим пропущенные значения


# In[15]:


cohort.drop(cohort.columns[[0, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 17]], axis = 1, inplace=True)
# Удалим ненужные колонки


# In[16]:


cohort.order_purchase_timestamp = pd.to_datetime(cohort.order_purchase_timestamp)
# Приведем колонку order_purchase_timestamp к формату dt


# In[17]:


cohort.order_purchase_timestamp.max()
# Посмтрим, когда был сделан последний доставленный заказ.
# 2018-08-29

# За "текущую" (отчетную) дату возьмем 2018-09-01 (след-ий месяц после последней покупки)


# In[18]:


cohort['report_date'] = '2018-09-01 23:59:59'
cohort.report_date = pd.to_datetime(cohort.report_date)
# Создадим колонку с одинаковым значением - отчетной датой, чтобы потом получить значение дельты.


# In[19]:


cohort['last_purch'] = cohort.groupby('customer_unique_id')['order_purchase_timestamp'].transform('max')
# Создадим колонку с датой последней покупки пользователя


# In[21]:


cohort['Recency'] = (cohort['report_date'] - cohort['last_purch']).dt.days
# Создадим колонку с количеством дней от даты последней покупки до отчетной даты.


# In[22]:


cohort_r = cohort.groupby('customer_unique_id', as_index = False) .agg({'Recency':'max'})
cohort_r
# Сделаем дф для дальнейшего подсчета квантилей для Recency


# In[23]:


cohort_r.Recency.min()
# Посмотрим минимальное и максимальное значение


# In[24]:


cohort_r.Recency.max()


# In[25]:


cohort_r.Recency.quantile(q=[0.2, 0.4, 0.6, 0.8])
# С помощью квантилей определим границы для 5 категорий
# 1   385 - 716
# 2   271 - 384
# 3   180 - 270
# 4   95 - 179
# 5     3 - 94


# In[27]:


def r(x):
    if x < 95:
        return 5
    elif 95 <= x < 180:
        return 4
    elif 180 <= x < 271:
        return 3
    elif 271 <= x < 385:
        return 2
    else:
        return 1
   # Напишем фунцию, чтобы заполнить новую колонку  R сегментами в зависимости от значений в колонке delta_purсh


# In[28]:


cohort['R'] = cohort['Recency'].apply(r)
# Создадим колонку со значениями R-сегмента


# In[29]:


cohort['Frequency'] = cohort.groupby('customer_unique_id')['order_id'].transform('nunique')
# Создадим новую колонку с количеством покупок (уникальных заказов) для каждого пользователя


# In[39]:


cohort.Frequency.min()
# Посмотрим минимальное и максимальное значение


# In[40]:


cohort.Frequency.max()


# In[41]:


cohort.Frequency.quantile(q=[0.2, 0.4, 0.6, 0.8])
# С помощью квантилей определим границы для 5 категорий


# In[42]:


cohort.groupby('Frequency') .agg({'customer_unique_id' : 'nunique'})
# Так как бОльшая часть пользователей сделала не более одного заказа, определим границы другим способом (вручную)


# In[43]:


def f(x):
    if x >= 7:
        return 5
    elif 5 <= x < 7:
        return 4
    elif 3 <= x < 5:
        return 3
    elif 1 < x < 3:
        return 2
    else:
        return 1
   # Напишем фунцию, чтобы заполнить новую колонку  F сегментами в зависимости от значений в колонке order_sum
# 1   1
# 2   2 
# 3   3 - 4
# 4   5 - 6
# 5   7 - 15


# In[44]:


cohort['F'] = cohort['Frequency'].apply(f)
# Создадим колонку со значениями F-сегмента


# In[45]:


cohort['Monetary'] = cohort.groupby('customer_unique_id')['price'].transform('sum')
# Создадим новую колонку с суммой покупок для каждого пользователя


# In[48]:


cohort_m = cohort.groupby('customer_unique_id', as_index = False) .agg({'Monetary':'max'})
cohort_m
# Сделаем дф для дальнейшего подсчета квантилей для Recency


# In[49]:


cohort_m.Monetary.min()


# In[50]:


cohort_m.Monetary.max()


# In[53]:


cohort_m.Monetary.quantile(q=[0.2, 0.4, 0.6, 0.8])
# С помощью квантилей определим границы для 5 категорий
# 1   0.85 - 39.89
# 2   39.9 - 69.8
# 3   69.9 - 109.8
# 4   109.9 - 179.8
# 5   179.9 - 13440.00


# In[54]:


def m(x):
    if x >= 179.9:
        return 5
    elif 109.9 <= x < 179.9:
        return 4
    elif 69.9 <= x < 109.9:
        return 3
    elif 39.9 <= x < 69.9:
        return 2
    else:
        return 1
   # Напишем фунцию, чтобы заполнить новую колонку  M сегментами в зависимости от значений в колонке price_sum


# In[55]:


cohort['M'] = cohort['Monetary'].apply(m)
# Создадим колонку со значениями M-сегмента


# In[56]:


cohort['RFM'] = cohort['R'].map(str)+cohort['F'].map(str)+cohort['M'].map(str)
# Создадим колонку RFM


# In[57]:


def s(x):
    if x == '555' or x == '554' or x == '544' or x == '545' or x == '454' or x == '455' or x == '445':   
        return 'Чемпионы'
    if x == '543' or x == '444' or x == '435' or x == '355' or x == '354' or x == '345' or x == '344' or x == '335':
        return 'Постоянные'
    if x == '553' or x == '551' or x == '552' or x == '541' or x == '542' or x == '533' or x == '532' or x == '531'     or x == '452' or x == '451' or x == '442' or x == '441' or x == '431' or x == '453' or x == '433' or x == '432'     or x == '423' or x == '353' or x == '352' or x == '351' or x == '342' or x == '341' or x == '333' or x == '323':
        return 'Потенциальные постоянные'
    if x == '512' or x == '511' or x == '422' or x == '421' or x == '412' or x == '411' or x == '311':
        return 'Новые'
    if x == '525' or x == '524' or x == '523' or x == '522' or x == '521' or x == '515'or x == '514' or x == '513'     or x == '425' or x == '424' or x == '413' or x == '414' or x == '415' or x == '315' or x == '314' or x == '313':
        return 'Перспективные'
    if x == '535' or x == '534' or x == '443' or x == '434' or x == '343' or x == '334' or x == '325' or x == '324':
        return 'Требуют внимания'
    if x == '331' or x == '321' or x == '312' or x == '221' or x == '213' or x == '231' or x == '241' or x == '251':
        return 'Почти уснули'
    if x == '155' or x == '154' or x == '144' or x == '214' or x == '215' or x == '115' or x == '114' or x == '113':
        return 'Нельзя потерять'
    if x == '255' or x == '254' or x == '245' or x == '244' or x == '253' or x == '252' or x == '243' or x == '242'     or x == '235' or x == '234' or x == '225' or x == '224' or x == '153' or x == '152' or x == '145' or x == '143'     or x == '142' or x == '135' or x == '134' or x == '133' or x == '125' or x == '124':
        return 'Зона риска'
    if x == '332' or x == '322' or x == '233' or x == '232' or x == '223' or x == '222' or x == '132' or x == '123'     or x == '122' or x == '212' or x == '211':
        return 'Уснувшие'
    if x == '111' or x == '112' or x == '121' or x == '131' or x == '141' or x == '151':
        return 'Потеряны'
# Напишем функцию, чтобы присвоить сегментам названия


# Чемпионы - Покупали недавно, заказывают часто и тратят больше всего.
# 
# Постоянные - Стабильно покупающие клиенты; часть из них может стать чемпионами. 
# 
# Потенциальные постоянные - Клиенты, которым немного не хватает частоты/суммы покупок, чтобы стать постоянными.
# 
# Новые - Купили что-то в первый раз, либо в первый раз за долгое время. 
# 
# Перспективные - Клиенты, которым нужно предлагать акции/программы лояльности.
# 
# Требуют внимания - Клиенты выше средней недавности, частоты и денежной стоимости.
# 
# Почти уснули - Клиенты в этом сегменте не совершали покупки в течение относительно длительного времени, 
# но не настолько, чтобы быть недоступными.
# 
# Нельзя потерять	- Клиенты, покупающие много и часто, но переставшие это делать. 
# 
# Зона риска - Клиенты, похожие на сегмент «Не можем терять», но ее участники недавно совершили покупку.
# 
# Уснувшие - Клиенты в этом сегменте не совершали покупки в течение  длительного времени
# 
# Потеряны - Этот сегмент имеет самый низкий приоритет.

# In[58]:


cohort['Seg'] = cohort['RFM'].apply(s)
# Создадим колонку с названиями сегментов


# In[59]:


rfm = cohort.groupby(['customer_unique_id', 'Seg'] , as_index = False) .agg({'RFM' : 'max'})
rfm
# Создадим дф с ун-ми значениями customer_unique_id	 и соостветствующими им Seg и RFM (т.к. RFM одинаковый, в агг-ей ф-ии применяю max)


# In[60]:


rfm_group = rfm.groupby('Seg', as_index = False) .agg({'customer_unique_id' : 'count'})
rfm_group
# Посмотрим распределение количества пользователей по сегментам.


# In[61]:


sns.set(rc={'figure.figsize':(20.14,15.35)})
sns.barplot(data=rfm_group, x="Seg", y = 'customer_unique_id')
# Посмотрим распределение клиентов по сегментам на барплоте. 


# # БОльшая часть клиентов - это перспективные, новые и требующие возврата/удержания (нельзя потерять).

# In[ ]:





# In[ ]:




