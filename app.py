from flask import Flask, render_template, request
import pandas as pd
import os
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('main.html')


@app.route('/statistics')
def statistics_page():
    return render_template('statistics.html')


@app.route('/result')
def result():
    return render_template('result.html')


@app.route('/process_statistics', methods=['GET', 'POST'])
def tatistics_route():
    if request.method == 'POST':
        salary = int(request.form.get('salary'))
        way = float(request.form.get('way'))
        homework = int(request.form.get('homework'))
        age = int(request.form.get('age'))
        omissions = int(request.form.get('omissions'))
        sex = float(request.form.get('sex'))
        mood = float(request.form.get('mood'))
        type_life = float(request.form.get('type_life'))

        data = {
            'salary': [salary],
            'way': [way],
            'homework': [homework],
            'age': [age],
            'omissions': [omissions],
            'sex': [sex],
            'mood': [age],
            'type_life': [type_life],
        }
        dfs = pd.DataFrame(data)

        #get_ipython().system(' pip install lightgbm')


        # In[23]:

        way_0 = r"C:\Users\Газиз\PycharmProjects\Statistics\data\Успеваемость_студентов_Ответы_на_форму_1.csv"
        way_1 = 'https://docs.google.com/spreadsheets/d/1VYHGD0hU1W4dKUBT04TE5YyLFwI3mqfa947FZczxIPs/edit#gid=1396735516'
        if os.path.exists(way_0):
            df = pd.read_csv(way_0)
        elif os.path.exists(way_1):
            df = pd.read_csv(way_1)
        else:
            print('Не удалось загрузить данные (O_o)')

        # In[25]:

        df = df.rename(
            columns={'На каком курсе вы обучаетесь?': 'course',
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

        # In[28]:

        df.drop(['время',
                 'course',
                 'sport',
                 'job',
                 'relatives',
                 'type_study',
                 'exam'
                 ], axis=1, inplace=True)

        # In[33]:

        df.info()

        # In[34]:

        condition = (df['score'] == '0') | (df['score'] == '-')
        df.score = df.score.replace(',', '.', regex=True)
        median_value = df.loc[df['age'] == 19, 'score'].median()
        df.loc[condition, 'score'] = median_value

        # In[35]:

        df.score.fillna(df.score.median(), inplace=True)

        # In[36]:

        df.loc[df['homework'] == '5-10', 'homework'] = 7
        df.loc[df['homework'] == '5-6', 'homework'] = 6

        # In[37]:

        df = df[df.salary < 40000]

        # In[38]:

        df.score = df.score.astype('float')

        # In[39]:

        df.homework = df.homework.astype('int')

        # In[40]:

        df.way.replace(',', '.', regex=True, inplace=True)
        df.way = df.way.astype('float')

        # In[41]:

        df = df[df.way < 10.0]

        # In[42]:

        features = df.drop(['score'], axis=1)
        target = df['score']

        # In[43]:

        features_train, features_test, target_train, target_test = train_test_split(
            features, target, test_size=0.25, random_state=4212
        )

        # In[44]:

        print(f'features_train.shape:  {features_train.shape}')
        print(f'features_test.shape: {features_test.shape}')
        print(f'target_train.shape: {target_train.shape}')
        print(f'target_test.shape: {target_test.shape}')

        # In[46]:

        df.info()

        # In[47]:

        categorical_features = ['sex', 'type_life', 'mood']
        features_train[categorical_features] = features_train[categorical_features].astype(str)
        features_test[categorical_features] = features_test[categorical_features].astype(str)

        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        enc.fit(features_train[categorical_features])

        features_train[categorical_features] = enc.transform(features_train[categorical_features])
        features_test[categorical_features] = enc.transform(features_test[categorical_features])

        # In[49]:

        final_model_lgb = lgb.LGBMRegressor(
            learning_rate=0.1,
            max_depth=1,
            n_estimators=300,
            num_leaves=10
        )

        final_model_lgb.fit(features_train, target_train)

        predictions = final_model_lgb.predict(dfs)

        return render_template('result.html', predictions=predictions)

    return render_template('statistics.html')


if __name__ == '__main__':
    app.run(debug=True)

# ruD6Eqeu72PtrRz5CyCP