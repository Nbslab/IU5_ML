import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


@st.cache
def load_data():
    data = pd.read_csv('data/wind_dataset.csv')
    return data


st.header('Вывод данных и графиков')

data_load_state = st.text('Загрузка данных...')
data = load_data()
data_load_state.text('Данные загружены!')

st.subheader('Первые 5 значений')
st.write(data.head())

if st.checkbox('Показать все данные'):
    st.subheader('Данные')
    st.write(data)

st.subheader('Скрипичные диаграммы для числовых колонок')
for col in ['WIND', 'IND', 'RAIN', 'IND.1', 'T.MAX', 'IND.2', 'T.MIN', 'T.MIN.G']:
    fig1 = plt.figure(figsize=(7, 5))
    ax = sns.violinplot(x=data[col])
    st.pyplot(fig1)

st.subheader('Тепловая карта корреляции')
fig2 = plt.figure(figsize=(7, 5))
ax = sns.heatmap(data.corr(), annot=True, fmt='.1f')
st.pyplot(fig2)
