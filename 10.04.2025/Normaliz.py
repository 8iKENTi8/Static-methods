import pandas as pd

# Загружаем данные с листа "Квартиры2"
df = pd.read_excel("flats_all_10_04_2025.xlsx",sheet_name="нормализация2")

# Нормализация всех числовых столбцов от 0 до 1
def normalize_columns(df):
    # Выбираем все числовые колонки
    numeric_columns = df.select_dtypes(include=["number"]).columns
    
    # Применяем нормализацию Min-Max для каждой числовой колонки
    for column in numeric_columns:
        min_value = df[column].min()
        max_value = df[column].max()
        # Нормализуем колонку
        df[column] = (df[column] - min_value) / (max_value - min_value)
    
    return df

# Применяем нормализацию
df_normalized = normalize_columns(df)

# Сохраняем нормализованные данные в новый Excel файл
df_normalized.to_excel("flats_normalized.xlsx", index=False)

print(f"✅ Нормализовано {len(df_normalized)} объявлений в flats_normalized.xlsx")
