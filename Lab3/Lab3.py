# датасет https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/data

import pandas as pd
import os
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

print(os.getcwd())

# Читаем Excel файл
data = pd.read_excel('dataset.xlsx')

print("Типы данных в датасете:")
print(data.dtypes)

# region 1 этап

print("\n=== 1 ЭТАП: Преобразование категориальных данных ===")

# Преобразуем 'Yes'/'No' в 1/0
data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

print(data.head())

# endregion


# region 2 этап - удаление выбросов по z-score

print("\n=== 2 ЭТАП: Удаление выбросов по z-score ===")

# Размер данных до удаления выбросов
print(f"Размер данных до удаления выбросов: {data.shape}")

# Считаем z-оценки для всех числовых столбцов
z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))

# Отбираем только те строки, где для всех признаков z-оценка меньше 3
data = data[(z_scores < 3).all(axis=1)]

# Размер данных после удаления выбросов
print(f"Размер данных после удаления выбросов: {data.shape}")

print(data.head())

# endregion


# region 3 этап - нормализация данных MinMaxScaler

print("\n=== 3 ЭТАП: Нормализация данных MinMaxScaler ===")

# Выбираем числовые колонки, кроме бинарного 'Extracurricular Activities'
num_cols = data.select_dtypes(include=['float', 'int']).columns.difference(['Extracurricular Activities'])

scaler = MinMaxScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# Проверяем результат нормализации
print("Данные после нормализации:")
print(data.head())

# endregion
