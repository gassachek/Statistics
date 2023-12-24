#!/usr/bin/env python
# coding: utf-8

# # Описание проекта

#     Был произведен сбор данных студентов с разных курсов. 
#     На основе этих данных будет построена модель, которая не только предоставляет прогнозы успеха, но также осуществляет анализ корреляции между оценками и различными признаками, такими как активность студентов, посещаемость лекций, использование учебных материалов и многое другое. 
#     Это позволяет студентам и учебным заведениям лучше понимать, какие факторы влияют на их успех и как они могут улучшить свои академические результаты.

# # План выполнения проекта

# 1) Подготовка данных
# 2) Анализ данных   
# 3) Проверить наличие аномалий и выбросов
# 4) Создать дополнительные параметры при необходимости
# 5) Очистить данные от признаков, которые не несут в себе никакой ценности 
# 6) Выполнить статистический анализ 
# 7) Представить визуализацию для статистического анализа
# 8) Выполнить масштабирование данных, регуляризацию данных и нормализацию  
# 9) Проверить данные для подготовки модели 
# 10) Создать модели и тестирование их 
# 11) Выбрать модель на основе лучшего значения метрики 
# 12) Выполнить тестирование модели на тестовых данных 
# 13) Визуализировать результат 
# 14) Подготовить модель к дальнейшему интегрированию в сайт

# # Цель проекта 

# Для предсказания средней оценки студентов разработать модель машинного обучения, основная метрика для оценки модели будет MAE (Mean Absolute Error), дополнительные метрики: MSE (Mean Squared Error), Коэффициент детерминации (R-squared)

# ## Подготовка данных 

# Импортируем все необходимые библиотеки

# In[1]:


get_ipython().system(' pip install lightgbm')
get_ipython().system(' pip install yellowbrick')
get_ipython().system(' pip install catboost')


# In[66]:


import pandas as pd

import os

import numpy as np

import plotly.express as px

import lightgbm as lgb

import pickle

import seaborn as sns

import matplotlib.pyplot as plt

from yellowbrick.classifier import ROCAUC

from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.compose import ColumnTransformer
from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[3]:


way_0 = r"C:\Users\kil_a\OneDrive\Рабочий стол\ГУАП\Проект\Успеваемость студентов - Ответы на форму (1).csv"
way_1 = 'https://docs.google.com/spreadsheets/d/1VYHGD0hU1W4dKUBT04TE5YyLFwI3mqfa947FZczxIPs/edit#gid=1396735516'
if os.path.exists(way_0):
    df = pd.read_csv(way_0)
elif os.path.exists(way_1):
    df = pd.read_csv(way_1)
else:
    print('Не удалось загрузить данные (O_o)')


# ## Анализ данных

# In[4]:


def check(data):
    """
    Функция для просмотра общих сведений по данным, такие как: 
    1) сводная статистика
    2) размерность данных 
    3) названия признаков (наименования колонок)
    """
    print(data.describe())
    print('\n---------------------------------------------')
    print('Используемые данные размерностью: ', data.shape)
    print('\n---------------------------------------------')
    print('Признаки, использующиеся для построения модели: ')
    for col in data.columns:
        print(col)


# In[5]:


check(df)


# In[6]:


def pass_value_barh(df):
    """
    Функция для наглядного предстваления пропусков по каждому столбцу 
    """
    try:
        (
            (df.isna().sum())
            .to_frame()
            .rename(columns = {0:'space'})
            .query('space > 0')
            .sort_values(by = 'space', ascending = True)
            .plot(kind = 'barh', figsize = (19,6), rot = -5, legend = False, fontsize = 16)
            .set_title('Количество пропусков' + "\n", fontsize = 22, color = 'SteelBlue')    
        );    
    except:
        print('пропусков не осталось')


# In[7]:


pass_value_barh(df)


# In[8]:


df = df.drop('время', axis = 1)


# In[9]:


print(f'Количество дубликатов в данных: {df.duplicated().sum()}')


# In[10]:


df = df.rename(
    columns = {'На каком курсе вы обучаетесь?': 'course', 
               'Сколько вам лет?': 'age',
               'Занимаетесь ли вы каким-то спортом вне ВУЗа ?': 'sport',
               'Ваш пол': 'sex',
               'Сколько приблизительно пар вы пропускаете в неделю?': 'omissions',
               'Подрабатываете ли вы или работаете помимо ВУЗа?': 'job',
               'Сколько приблизительно часов в неделю вы уделяете на выполнения домашнего задания и закрепления информации, полученной в ВУЗе?': 'homework',
               'Где вы проживаете?': 'type_life',
               'Сколько вы зарабатываете с учетом стипендии?': 'salary',
               'Укажите количество родных братьев или сестер': 'relatives', 
               'На какой основе вы обучаетесь?': 'type_study',
               'Укажите ваш суммарный балл ЕГЭ': 'exam',
               'Сколько времени в часах занимает ваша дорога до ВУЗа?': 'way',
               'Считаете ли вы высшим образованием необходимым?': 'mood',
               'Ваша средняя оценка за предыдущий семестр': 'score'}
)


# In[11]:


df.sort_values(by = 'score')['score'].hist(figsize = (15, 8));
plt.title('Распределение средних оценко за семестр по ученикам')
plt.xlabel('Средняя оценка за семестр')
plt.ylabel('Количество студентов')
plt.show()


# In[12]:


df.course.unique()


# In[13]:


df[df.score == '-']


# In[14]:


df[df.score == '0']


# In[15]:


condition = (df['score'] == '0') | (df['score'] == '-')
df.score = df.score.replace(',', '.', regex=True)
median_value = df.loc[df['course'] == '2 курс (бакалавриата)', 'score'].median()
df.loc[condition, 'score'] = median_value


# In[16]:


df.score.fillna(df.score.median(), inplace = True)


# In[17]:


df.exam.fillna(df.exam.median(), inplace = True)


# In[18]:


df.info()


# In[19]:


df.loc[df['homework'] == '5-10', 'homework'] = 7
df.loc[df['homework'] == '5-6', 'homework'] = 6


# In[20]:


df.salary.describe()


# In[21]:


df = df[df.salary < 40000]


# In[22]:


df.salary.hist(figsize = (12, 6))
plt.title('Распределение зарплаты студентов')
plt.xlabel('Уровень зарплаты студента')
plt.ylabel('Количество студентов')
plt.show()
box = px.box(df.salary)
box.show()


# In[23]:


df.exam.hist(figsize = (12, 6))
plt.title('Распределение баллов ЕГЭ среди студентов')
plt.xlabel('Балл по ЕГЭ')
plt.ylabel('Количество студентов')
plt.show()
box = px.box(df.exam)
box.show()


# In[24]:


df.score = df.score.astype('float')


# In[25]:


df.info()


# In[26]:


df.homework = df.homework.astype('int')


# In[27]:


df.exam = df.exam.astype('int')


# In[28]:


df.way.unique()


# In[29]:


df.way.replace(',', '.', regex = True, inplace = True)
df.way = df.way.astype('float')


# In[30]:


df.way.describe()


# In[31]:


df = df[df.way < 10.0]


# In[32]:


df.way.hist(figsize = (12, 6))
plt.title('Распределение времени пути до ВУЗа')
plt.xlabel('Потраченное время в часах')
plt.ylabel('Количество студентов')
plt.show()
box = px.box(df.way)
box.show()


# In[33]:


features = df.drop(['score'], axis = 1)
target = df['score']


# In[34]:


features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size = 0.25, random_state = 4212
)


# In[35]:


print(f'features_train.shape:  {features_train.shape}')
print(f'features_test.shape: {features_test.shape}')
print(f'target_train.shape: {target_train.shape}')
print(f'target_test.shape: {target_test.shape}')


# In[36]:


df.info()


# In[37]:


categorical_features = ['course', 'sport', 'sex', 'job', 'type_life', 'type_study', 'mood']
features_train[categorical_features] = features_train[categorical_features].astype(str)
features_test[categorical_features] = features_test[categorical_features].astype(str)

enc = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)
enc.fit(features_train[categorical_features])

features_train[categorical_features] = enc.transform(features_train[categorical_features])
features_test[categorical_features] = enc.transform(features_test[categorical_features])


# In[38]:


(features_train == -1).sum().sum()


# In[39]:


(features_test == -1).sum().sum()


# In[40]:


(target_train == -1).sum().sum()


# In[41]:


(target_test == -1).sum().sum()


# In[42]:


corr = features_train.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[43]:


features_train = features_train.drop(['exam',
                                      'job',
                                      'sport',
                                      'course',
                                      'type_study',
                                      'relatives'
                                     ], axis = 1)
features_test = features_test.drop(['exam',
                                    'job',
                                    'sport',
                                    'course',
                                    'type_study',
                                    'relatives'
                                   ], axis = 1)


# In[44]:


model_dr = DummyRegressor()
model_dr.fit(features_train, target_train)
predictions = model_dr.predict(features_test)
r2 = r2_score(target_test, predictions)
rmse = mean_squared_error(target_test, predictions)
print(f'Значение R2 на тестовых данных: {abs(r2):.2f}')
print(f'Значение MSE на тестовых данных {rmse:.2f}')


# In[45]:


model_dtr = DecisionTreeRegressor(random_state = 42)

parameters = {
    'max_depth' : [x for x in range(1, 8)],
    'min_samples_split' : [x for x in np.arange(0.1, 1, 0.1)],
    'min_samples_leaf' : [x for x in range(1, 5)]
}

scoring_metrics = {
    'r2': 'r2',
    'mse': make_scorer(mean_squared_error, greater_is_better=False)
}

grid_model_dtr = GridSearchCV(
    model_dtr,
    parameters,
    cv=3,
    scoring=scoring_metrics,
    refit='r2'  
)

grid_model_dtr.fit(features_train, target_train)

best_r2 = grid_model_dtr.cv_results_['mean_test_r2'][grid_model_dtr.best_index_]
best_mse = abs(grid_model_dtr.cv_results_['mean_test_mse'][grid_model_dtr.best_index_])

print(f'Лучшие параметры: {grid_model_dtr.best_params_}')
print(f'Значение метрики R2: {abs(best_r2):.2f}')
print(f'Значение метрики MSE: {abs(best_mse):.2f}')


# In[46]:


model_rfr = RandomForestRegressor(random_state=12345)
parameters = { 'max_depth': [x for x in range(1, 15)],
               'n_estimators': [x for x in range(10, 71, 10)]
             }
scoring_metrics = {
    'r2': 'r2',
    'mse': make_scorer(mean_squared_error, greater_is_better=False)
}

grid_model_rfr = GridSearchCV(
    model_rfr,
    parameters,
    cv=3,
    scoring=scoring_metrics,
    refit='r2'  
)

grid_model_rfr.fit(features_train, target_train)

best_r2 = grid_model_rfr.cv_results_['mean_test_r2'][grid_model_rfr.best_index_]
best_mse = abs(grid_model_rfr.cv_results_['mean_test_mse'][grid_model_rfr.best_index_])

print(f'Лучшие параметры: {grid_model_rfr.best_params_}')
print(f'Значение метрики R2: {abs(best_r2):.2f}')
print(f'Значение метрики MSE: {abs(best_mse):.2f}')


# In[47]:


model_lgb = lgb.LGBMRegressor(num_leaves = 31, learning_rate = 0.01)
parameters = {'n_estimators': [x for x in range(300, 331, 10)],
              'max_depth': [x for x in range(1, 5, 2)],
              'num_leaves': [x for x in range(10, 21, 2)]
             }
scoring_metrics = {
    'r2': 'r2',
    'mse': make_scorer(mean_squared_error, greater_is_better=False)
}

grid_model_lgb = GridSearchCV(
    model_lgb,
    parameters,
    cv=3,
    scoring=scoring_metrics,
    refit='r2'  
)

grid_model_lgb.fit(features_train, target_train)

best_r2 = grid_model_lgb.cv_results_['mean_test_r2'][grid_model_lgb.best_index_]
best_mse = abs(grid_model_lgb.cv_results_['mean_test_mse'][grid_model_lgb.best_index_])

print(f'Лучшие параметры: {grid_model_lgb.best_params_}')
print(f'Значение метрики R2: {abs(best_r2):.2f}')
print(f'Значение метрики MSE: {abs(best_mse):.2f}')


# In[49]:


scaler = StandardScaler()
int_columns = ['age', 'omissions', 'homework', 'salary', 'way']
features_train[int_columns] = scaler.fit_transform(features_train[int_columns])
feat_test = features_test.copy()
features = scaler.transform(features_test[int_columns])
features_test[int_columns] = features


# In[50]:


features_train


# In[51]:


model_dr = DummyRegressor()
model_dr.fit(features_train, target_train)
predictions = model_dr.predict(features_test)
r2 = r2_score(target_test, predictions)
rmse = mean_squared_error(target_test, predictions)
print(f'Значение R2 на тестовых данных: {abs(r2):.2f}')
print(f'Значение MSE на тестовых данных {rmse:.2f}')


# In[52]:


model_lgb = lgb.LGBMRegressor()
parameters = {'n_estimators': [x for x in range(300, 331, 10)],
              'max_depth': [x for x in range(1, 5, 2)],
              'num_leaves': [x for x in range(10, 21, 2)], 
              'learning_rate': [x for x in np.arange(0.1, 1, 0.1)]
             }
scoring_metrics = {
    'r2': 'r2',
    'mse': make_scorer(mean_squared_error, greater_is_better=False)
}

grid_model_lgb = GridSearchCV(
    model_lgb,
    parameters,
    cv=3,
    scoring=scoring_metrics,
    refit='r2'  
)

grid_model_lgb.fit(features_train, target_train)

best_r2 = grid_model_lgb.cv_results_['mean_test_r2'][grid_model_lgb.best_index_]
best_mse = abs(grid_model_lgb.cv_results_['mean_test_mse'][grid_model_lgb.best_index_])

print(f'Лучшие параметры: {grid_model_lgb.best_params_}')
print(f'Значение метрики R2: {abs(best_r2):.2f}')
print(f'Значение метрики MSE: {abs(best_mse):.2f}')


# In[53]:


final_model_rfr = RandomForestRegressor(
    max_depth = 1, 
    n_estimators = 10,
    random_state = 12345
)

final_model_rfr.fit(features_train, target_train)

predictions = final_model_rfr.predict(features_test)
r2 = r2_score(target_test, predictions)
mse = mean_squared_error(target_test, predictions)

print(f'Значение метрики R2 на тестовых данных {abs(r2):.2f}')
print(f'Значение метрики MSE на тестовых данных {abs(mse):.2f}')


# In[54]:


final_model_lgb = lgb.LGBMRegressor(
    learning_rate =  0.1,
    max_depth = 1, 
    n_estimators = 300, 
    num_leaves = 10
)

final_model_lgb.fit(features_train, target_train)

predictions = final_model_lgb.predict(features_test)
r2 = r2_score(target_test, predictions)
mse = mean_squared_error(target_test, predictions)

print(f'Значение метрики R2 на тестовых данных {abs(r2):.2f}')
print(f'Значение метрики MSE на тестовых данных {abs(mse):.2f}')


# In[55]:


plt.scatter(target_test, predictions)
plt.plot([min(target_test), max(target_test)], [min(target_test), max(target_test)], linestyle='--', color='red')
plt.xlabel('Фактические значения')
plt.ylabel('Предсказанные значения')
plt.title('График рассеяния между фактическими и предсказанными значениями')
plt.show()


# In[56]:


plt.plot(np.arange(len(target_test)), target_test, marker='o', linestyle='-', color='blue', label='Фактические значения')
plt.plot(np.arange(len(predictions)), predictions, marker='o', linestyle='-', color='red', label='Предсказанные значения')

plt.xlabel('Номер наблюдения')
plt.ylabel('Значения')
plt.title('График фактических и предсказанных значений с линиями')
plt.legend()
plt.show()


# In[57]:


feature_names = features_test.columns
feature_importances = final_model_lgb.feature_importances_
feature_importance_series = pd.Series(feature_importances, index = feature_names)
feature_importance_series.sort_values(ascending=True).plot.barh(grid=True)
plt.title('Важность признаков для обучения')
plt.show()


# In[58]:


df.info()


# In[59]:


data = df[['score', 'type_study']]

grouped_data = data.groupby('type_study')['score'].mean().reset_index()

colors = plt.cm.get_cmap('tab10', len(grouped_data['type_study']))
bars = plt.bar(grouped_data['type_study'], grouped_data['score'], color=colors(range(len(grouped_data['type_study']))))

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom')

plt.xlabel('Тип обучения')
plt.ylabel('Средний балл')
plt.ylim(0, 6)
plt.title('Средний балл в зависимости от типа обучения')

plt.xticks(rotation=45, ha='right')

plt.show()


# In[60]:


data = df[['score', 'course']]

grouped_data = data.groupby('course')['score'].mean().reset_index()

colors = plt.cm.get_cmap('tab10', len(grouped_data['course']))
bars = plt.bar(grouped_data['course'], grouped_data['score'], color=colors(range(len(grouped_data['course']))))

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom')

plt.xlabel('Курс')
plt.ylabel('Средний балл')
plt.ylim(0, 6)
plt.title('Средний балл в зависимости от курса')

plt.xticks(rotation=45, ha='right')

plt.show()


# In[61]:


data = df[['score', 'sport']]

grouped_data = data.groupby('sport')['score'].mean().reset_index()

colors = plt.cm.get_cmap('tab10', len(grouped_data['sport']))
bars = plt.bar(grouped_data['sport'], grouped_data['score'], color=colors(range(len(grouped_data['sport']))))

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom')

plt.xlabel('Вид спорта')
plt.ylabel('Средний балл')
plt.ylim(0, 6)
plt.title('Средний балл в зависимости от вида спорта')
plt.show()


# In[62]:


data = df[['score', 'salary']]

plt.figure(figsize=(10, 6))
plt.scatter(data['salary'], data['score'], c=data['score'], cmap='viridis', alpha=0.7)

plt.xlabel('Зарплата')
plt.ylabel('Средний балл')
plt.title('Диаграмма рассеяния между зарплатой и средним баллом ')
plt.colorbar(label='Баллы')
plt.show()


# In[63]:


data = df[['score', 'salary']]

average_score_by_salary = data.groupby('salary')['score'].mean().reset_index()

plt.figure(figsize=(10, 6))
plt.plot(average_score_by_salary['salary'], average_score_by_salary['score'], marker='o', color='green', linestyle='-')

plt.xlabel('Зарплата')
plt.ylabel('Средний балл')
plt.title('Зависимость между зарплатой и средним баллом по итогу сессии')
plt.show()


# In[64]:


data = df[['score', 'salary']]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='salary', y='score', data=data, hue='salary', palette='viridis', s=50)

plt.xlabel('Зарплата')
plt.ylabel('Средний балл')
plt.title('Связь между зарплатой и cредним баллом по итогу сессии')
plt.legend(title='Зарплата')
plt.show()


# In[65]:


data = df[['score', 'type_life']]

grouped_data = data.groupby('type_life')['score'].mean().reset_index()

colors = plt.cm.get_cmap('tab10', len(grouped_data['type_life']))
bars = plt.bar(grouped_data['type_life'], grouped_data['score'], color=colors(range(len(grouped_data['type_life']))))

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, round(yval, 2), ha='center', va='bottom')

plt.xlabel('Тип жизни')
plt.ylabel('Средний балл')
plt.ylim(0, 6)
plt.title('Средний балл в зависимости от вида проживания')

plt.xticks(rotation=45, ha='right')

plt.show()


# In[ ]:




