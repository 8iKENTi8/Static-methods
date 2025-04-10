import pandas as pd
import numpy as np

# Загружаем данные с листа "Квартиры2"
df = pd.read_excel("flats_all_10_04_2025.xlsx", sheet_name="нормализация2")

# Применяем экспоненциальную функцию к колонке "метро" (5-я колонка)
def apply_exponential(df, column_index):
    # Применяем экспоненциальную формулу к указанной колонке
    df.iloc[:, column_index] = np.exp(df.iloc[:, column_index])
    
    return df

# Применяем экспоненциальную формулу к колонке "метро" (5-й индекс, 0-based)
df_with_exp_metro = apply_exponential(df, 4)  # Индекс 4 для 5-й колонки

# Сохраняем данные в новый Excel файл
df_with_exp_metro.to_excel("flats_with_exp_metro.xlsx", index=False)

print(f"✅ Применена экспоненциальная функция к колонке 'метро'. Данные сохранены в flats_with_exp_metro.xlsx")
