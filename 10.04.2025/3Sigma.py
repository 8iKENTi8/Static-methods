import pandas as pd

# Загружаем данные из Excel
df = pd.read_excel("flats_all_10_04_2025.xlsx", sheet_name="Квартиры2")

# Функция для применения правила 3 сигм
def apply_3sigma_rule(df, column):
    # Вычисляем среднее и стандартное отклонение
    mean = df[column].mean()
    std_dev = df[column].std()
    
    # Вычисляем верхний и нижний пределы
    upper_limit = mean + 3 * std_dev
    lower_limit = mean - 3 * std_dev
    
    # Фильтруем данные, оставляем только те, которые находятся в пределах этих границ
    filtered_df = df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]
    
    return filtered_df

# Применяем правило 3 сигм для каждой колонки
df_filtered = df

df_filtered = apply_3sigma_rule(df_filtered, "время до метро")
df_filtered = apply_3sigma_rule(df_filtered, "floors_count")

# Сохраняем отфильтрованные данные в новый Excel файл
df_filtered.to_excel("flats_filtered_with_3sigma_all_columns.xlsx", index=False)

print(f"✅ Сохранено отфильтрованных {len(df_filtered)} объявлений в flats_filtered_with_3sigma_all_columns.xlsx")
