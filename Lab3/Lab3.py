# датасет https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/data

import pandas as pd
import os
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("Текущая директория:", os.getcwd())

# Читаем Excel файл
data = pd.read_excel('dataset.xlsx')

print("Типы данных в датасете:")
print(data.dtypes)

# region 1 этап - Преобразование категориальных данных

print("\n=== 1 ЭТАП: Преобразование категориальных данных ===")

# Преобразуем 'Yes'/'No' в 1/0
data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

print(data.head())

# endregion

# region 2 этап - Проверка и обработка пропусков

print("\n=== 2 ЭТАП: Проверка и обработка пропусков ===")

# Проверка пропусков
missing = data.isnull().sum()
print("Пропуски по столбцам:")
print(missing)

# Обработка пропусков — простой способ: числовые → median, категориальные (если будут) → mode
for col in data.columns:
    if data[col].isnull().sum() > 0:
        if data[col].dtype in ['float64', 'int64']:
            data[col].fillna(data[col].median(), inplace=True)
        else:
            data[col].fillna(data[col].mode()[0], inplace=True)

print("Пропуски после заполнения:")
print(data.isnull().sum())

# endregion

# region 3 этап - Удаление выбросов по z-score

print("\n=== 3 ЭТАП: Удаление выбросов по z-score ===")

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

# region 4 этап - Нормализация данных MinMaxScaler

print("\n=== 4 ЭТАП: Нормализация данных MinMaxScaler ===")

# Выбираем числовые колонки, кроме бинарного 'Extracurricular Activities'
num_cols = data.select_dtypes(include=['float64', 'int64']).columns.difference(['Extracurricular Activities'])

scaler = MinMaxScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# Проверяем результат нормализации
print("Данные после нормализации:")
print(data.head())

# endregion

# region 5 этап - Анализ данных (EDA)

print("\n=== 5 ЭТАП: EDA - Анализ распределений, корреляций и выбросов ===")

# Распределение признаков
data.hist(bins=15, figsize=(15,10))
plt.suptitle('Распределение признаков')
plt.show()

# Корреляционная матрица
plt.figure(figsize=(10,7))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица')
plt.show()

# Boxplot для проверки выбросов
plt.figure(figsize=(12,6))
sns.boxplot(data=data[num_cols])
plt.title('Boxplot числовых признаков')
plt.xticks(rotation=45)
plt.show()

# endregion

# region 6 этап - Разделение данных и обучение модели

print("\n=== 6 ЭТАП: Разделение выборки и обучение модели ===")



# Разделяем признаки и целевую переменную
X = data.drop(columns=['Performance Index'])
y = data['Performance Index']

# Разделяем на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказания
y_pred = model.predict(X_test)

# Оценка качества
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Среднеквадратичная ошибка (MSE): {mse:.4f}")
print(f"Коэффициент детерминации (R²): {r2:.4f}")

# Визуализация
plt.scatter(y_test, y_pred)
plt.xlabel("Фактический Performance Index")
plt.ylabel("Предсказанный Performance Index")
plt.title("Фактическое vs Предсказанное")
plt.plot([0,1],[0,1], 'r--')
plt.grid()
plt.show()

# endregion