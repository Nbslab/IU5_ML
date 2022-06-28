import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def date_to_unix(row):
    return (pd.Timestamp(row['DATE']).timestamp())


@st.cache
def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv('data/wind_dataset.csv')
    return data


@st.cache
def preprocess_data(data_in):
    '''
    Обработка пропусков, масштабирование признаков, функция возвращает X и y для кросс-валидации
    '''
    data_out = data_in.copy()
    data_out['DATE'] = data_out.apply(date_to_unix, axis=1)
    data_out['T.MAX'] = data_out['T.MAX'].fillna(data_out['T.MAX'].median())
    data_out['T.MIN'] = data_out['T.MIN'].fillna(data_out['T.MIN'].median())
    data_out['T.MIN.G'] = data_out['T.MIN.G'].fillna(data_out['T.MIN.G'].median())
    data_out = data_out.dropna(subset=['IND.1', 'IND.2']).reset_index(drop=True)
    # Числовые колонки для масштабирования
    scale_cols = ['DATE', 'IND', 'RAIN', 'IND.1', 'T.MAX', 'IND.2', 'T.MIN', 'T.MIN.G']
    new_cols = []
    sc1 = MinMaxScaler()
    sc1_data = sc1.fit_transform(data_out[scale_cols])
    for i in range(len(scale_cols)):
        col = scale_cols[i]
        new_col_name = col + '_scaled'
        new_cols.append(new_col_name)
        data_out[new_col_name] = sc1_data[:, i]

    temp_X = data_out.drop(['WIND', 'T.MIN.G', 'IND.1', 'T.MIN'], axis=1)
    # temp_X = data_out[new_cols]
    temp_y = data_out['WIND']
    # Чтобы в тесте получилось низкое качество используем только 0,5% данных для обучения
    X_train, X_test, y_train, y_test = train_test_split(temp_X, temp_y, train_size=0.75, random_state=1)
    return X_train, X_test, y_train, y_test


# Модели
models_list = ['LinR', 'KNN_5', 'Tree', 'RF', 'GB']
clas_models = {'LinR': LinearRegression(),
               'KNN_5': KNeighborsRegressor(n_neighbors=5),
               'Tree': DecisionTreeRegressor(),
               'RF': RandomForestRegressor(),
               'GB': GradientBoostingRegressor()}


@st.cache(suppress_st_warning=True)
def print_models(models_select, X_train, X_test, y_train, y_test):
    current_models_list = []
    mae_list = []
    mse_list = []
    for model_name in models_select:
        model = clas_models[model_name]
        model.fit(X_train, y_train)
        # Предсказание значений
        Y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test.values, Y_pred)
        mse = mean_squared_error(y_test.values, Y_pred)
        current_models_list.append(model_name)
        mae_list.append(mae)
        mse_list.append(mse)

    if len(mae_list) > 0:
        temp_d = {'MAE': mae_list}
        temp_df = pd.DataFrame(data=temp_d, index=current_models_list)
        st.bar_chart(temp_df)

    if len(mse_list) > 0:
        temp_d2 = {'MSE': mse_list}
        temp_df2 = pd.DataFrame(data=temp_d2, index=current_models_list)
        st.bar_chart(temp_df2)


st.sidebar.header('Модели машинного обучения')
models_select = st.sidebar.multiselect('Выберите модели', models_list)

data = load_data()
X_train, X_test, y_train, y_test = preprocess_data(data)

st.header('Оценка качества моделей')
print_models(models_select, X_train, X_test, y_train, y_test)
