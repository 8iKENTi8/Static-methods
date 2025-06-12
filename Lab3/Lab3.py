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
import math
from sklearn.metrics import calinski_harabasz_score

# region метрики

def compute_distance_matrix(data, metric):
    """
    Строит матрицу расстояний для набора объектов.
    data — список точек: [[x1, y1, ...], [x2, y2, ...], ...]
    Возвращает квадратную матрицу расстояний.
    """
    n = len(data)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                if metric == 'euclidean':
                    dist = euclidean_distance2(data[i], data[j])
                if metric == 'manhattan':
                    dist = manhattan_distance(data[i], data[j])
                if metric == 'chebyshev':
                    dist = chebyshev_distance(data[i], data[j])
                if metric == 'minkowski':
                    dist = minkowski_distance(data[i], data[j], 2, 3)
                if metric == 'mahalanobis':
                    dist = mahalanobis_distance(data[i], data[j], cov_inv)
                if metric == 'spearman':
                    dist = spearman_rank_correlation(data[i], data[j])
                if metric == 'kendall':
                    dist = kendall_tau(data[i], data[j])
                if metric == 'pearson':
                    dist = pearson_correlation(data[i], data[j])
                matrix[i][j] = dist
            else:
                matrix[i][j] = 0.0  # расстояние до себя

    return matrix

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
def euclidean_distance2(p1, p2):
    return sum((a - b) ** 2 for a, b in zip(p1, p2))
def manhattan_distance(p1, p2):
    return sum(abs(a - b) for a, b in zip(p1, p2))
def chebyshev_distance(p1, p2):
    return max(abs(a - b) for a, b in zip(p1, p2))
def minkowski_distance(p1, p2, p, r):
    return sum(abs(a - b) ** p for a, b in zip(p1, p2)) ** (1 / r)
def mahalanobis_distance(p1, p2, cov_inv):
    delta = np.array(p1) - np.array(p2)
    return np.sqrt(np.dot(np.dot(delta.T, cov_inv), delta))
def spearman_rank_correlation(x, y):
    n = len(x)
    rank_x = {val: rank for rank, val in enumerate(sorted(x), 1)}
    rank_y = {val: rank for rank, val in enumerate(sorted(y), 1)}
    d_squared = sum((rank_x[a] - rank_y[b]) ** 2 for a, b in zip(x, y))
    return 1 - (6 * d_squared) / (n * (n**2 - 1))
def kendall_tau(x, y):
    n = len(x)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = x[i] - x[j]
            b = y[i] - y[j]
            if a * b > 0:
                concordant += 1
            elif a * b < 0:
                discordant += 1
    return (concordant - discordant) / (0.5 * n * (n - 1))
def pearson_correlation(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = sum((a - mean_x) ** 2 for a in x)
    den_y = sum((b - mean_y) ** 2 for b in y)
    return num / math.sqrt(den_x * den_y)

# endregion

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

# region 5 этап - Подвыборка 40% и матрица расстояний

print("\n=== 5 ЭТАП: Подвыборка 40% и матрица расстояний ===")

# Можно использовать все данные, они уже числовые
X = data

# Делаем подвыборку 40%
X = X.sample(frac=0.4, random_state=42).reset_index(drop=True)

# Считаем матрицу расстояний
X_t = compute_distance_matrix(X.values, 'euclidean')

# Показываем первые строки
print("Матрица расстояний (первые 5 строк):")
print(pd.DataFrame(X_t).head())

# endregion

# region 6 этап - Гистограмма расстояний

print("\n=== 6 ЭТАП: Гистограмма расстояний ===")

def build_distance_histogram(distance_matrix, bins=20):
    n = len(distance_matrix)
    distances = []
    
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(distance_matrix[i][j])

    # Определяем минимальное и максимальное расстояние
    min_d = min(distances)
    max_d = max(distances)
    step = (max_d - min_d) / bins

    histogram = [0] * bins

    for d in distances:
        index = int((d - min_d) / step)
        if index == bins:
            index -= 1  # включаем правую границу в последний столбец
        histogram[index] += 1

    # Вывод гистограммы в консоль и на график
    binss = []
    counts = []
    print("Гистограмма расстояний:")
    for i in range(bins):
        left = min_d + i * step
        right = left + step
        counts.append(histogram[i])
        binss.append(f"[{left:.2f}, {right:.2f})")

    for b, c in zip(binss, counts):
        print(f"{b}: {c}")

    plt.barh(binss, counts, color='skyblue')
    plt.xlabel('Количество')
    plt.ylabel('Интервалы')
    plt.tight_layout()
    plt.show()

build_distance_histogram(X_t)

# endregion

#region Region 7 - Вроцлавская таксономия (Р М СК НЗ)

import networkx as nx
import numpy as np

class WroclawTaxonomyClustering:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.n = len(distance_matrix)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.n))

    def cluster_with_threshold(self, threshold):
        self.graph.clear_edges()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.distance_matrix[i][j] < threshold:
                    self.graph.add_edge(i, j)
        components = list(nx.connected_components(self.graph))
        return [list(c) for c in components]

def compute_within_cluster_distance(distance_matrix, clusters):
    """
    Вычисляет сумму внутрикластерных квадратов отклонений по евклидову расстоянию
    на основе расстояний между объектами и центрами кластеров.
    """
    total = 0.0
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        # Находим центр кластера как среднее в исходном пространстве (данные X)
        center = np.mean(X.iloc[cluster].values, axis=0)
        for i in cluster:
            v = X.iloc[i].values - center
            total += np.dot(v, v)  # квадрат евклидова расстояния
    return total

def wroclaw_taxonomy_elbow(distance_matrix, thresholds):
    print("=== Вроцлавская таксономия: Р М СК НЗ ===")
    scores = []
    cluster_counts = []
    for t in thresholds:
        clustering = WroclawTaxonomyClustering(distance_matrix)
        clusters = clustering.cluster_with_threshold(t)
        cluster_counts.append(len(clusters))
        score = compute_within_cluster_distance(distance_matrix, clusters)
        scores.append(score)
        print(f"Порог: {t:.4f}, Кластеры: {len(clusters)}, Внутрикластерный разброс: {score:.4f}")

    plt.figure(figsize=(8,4))
    plt.plot(cluster_counts, scores, 'ro-')
    plt.xlabel('Число кластеров')
    plt.ylabel('Внутрикластерный разброс')
    plt.title('Метод локтя — Вроцлавская таксономия')
    plt.grid(True)
    plt.show()

    return scores

thresholds = np.linspace(0.01, 0.081, num=15)
wroclaw_taxonomy_elbow(X_t, thresholds[::-1])

#endregion


# region 8 этап - Индекс силуэта для кластеризации Вроцлавской таксономии

from sklearn.metrics import silhouette_score

def wroclaw_taxonomy_elbow_with_silhouette(distance_matrix, thresholds):
    print("=== Вроцлавская таксономия: Р М СК НЗ с индексом силуэта ===")
    scores = []
    silhouettes = []
    cluster_counts = []

    for t in thresholds:
        clustering = WroclawTaxonomyClustering(distance_matrix)
        clusters = clustering.cluster_with_threshold(t)
        cluster_counts.append(len(clusters))
        score = compute_within_cluster_distance(distance_matrix, clusters)

        # Преобразование кластеров в метки
        labels = np.zeros(len(distance_matrix), dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for index in cluster:
                labels[index] = cluster_id

        # Проверка количества уникальных меток
        n_clusters = len(set(labels))
        if n_clusters > 1:
            silhouette = silhouette_score(X.values, labels, metric='euclidean')
        else:
            silhouette = np.nan  # Или 0, если хотите

        scores.append(score)
        silhouettes.append(silhouette)

        print(f"Порог: {t:.4f}, Кластеры: {len(clusters)}, Внутрикластерный разброс: {score:.4f}, Silhouette: {silhouette}")

    # Фильтруем NaN для корректного построения графика
    filtered_counts = [c for c, s in zip(cluster_counts, silhouettes) if not np.isnan(s)]
    filtered_silhouettes = [s for s in silhouettes if not np.isnan(s)]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(cluster_counts, scores, 'ro-')
    plt.xlabel('Число кластеров')
    plt.ylabel('Внутрикластерный разброс')
    plt.title('Метод локтя — Вроцлавская таксономия')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(filtered_counts, filtered_silhouettes, 'bo-')
    plt.xlabel('Число кластеров')
    plt.ylabel('Индекс силуэта')
    plt.title('Индекс силуэта по числу кластеров')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return scores, silhouettes

# Запускаем с порогами
thresholds = np.linspace(0.01, 0.081, num=15)
wroclaw_taxonomy_elbow_with_silhouette(X_t, thresholds[::-1])

# endregion

# region 9 этап - Индекс Калински-Харабаша для кластеров Вроцлавской таксономии

def wroclaw_taxonomy_with_calinski(distance_matrix, thresholds):
    print("=== Вроцлавская таксономия с индексом Калински-Харабаша ===")
    ch_scores = []
    cluster_counts = []

    for t in thresholds:
        clustering = WroclawTaxonomyClustering(distance_matrix)
        clusters = clustering.cluster_with_threshold(t)
        cluster_counts.append(len(clusters))

        # Формируем метки для объектов из кластеров
        labels = np.zeros(len(distance_matrix), dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for index in cluster:
                labels[index] = cluster_id

        # Индекс Калински-Харабаша требует минимум 2 кластера
        if len(set(labels)) > 1:
            ch_score = calinski_harabasz_score(X.values, labels)
        else:
            ch_score = np.nan

        ch_scores.append(ch_score)
        print(f"Порог: {t:.4f}, Кластеры: {len(clusters)}, Индекс Калински-Харабаша: {ch_score}")

    # Фильтрация nan значений
    filtered_counts = [c for c, s in zip(cluster_counts, ch_scores) if not np.isnan(s)]
    filtered_scores = [s for s in ch_scores if not np.isnan(s)]

    plt.figure(figsize=(8,4))
    plt.plot(filtered_counts, filtered_scores, 'go-')
    plt.xlabel('Число кластеров')
    plt.ylabel('Индекс Калински-Харабаша')
    plt.title('Индекс Калински-Харабаша для Вроцлавской таксономии')
    plt.grid(True)
    plt.show()

    return ch_scores

# Запускаем с теми же порогами
thresholds = np.linspace(0.01, 0.081, num=15)
wroclaw_taxonomy_with_calinski(X_t, thresholds[::-1])

# endregion
