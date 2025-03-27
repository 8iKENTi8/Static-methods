import cianparser
import openpyxl

# Создаём парсер для Москвы
moscow_parser = cianparser.CianParser(location="Москва")

# Собираем данные (~500 записей)
data = moscow_parser.get_flats(
    deal_type="sale", 
    rooms=(1, 2, 3, 4, 5),  
    with_saving_csv=True,  
    additional_settings={
        "start_page": 1,
        "end_page": 20,
        "max_floor": 45,  # Ограничение по максимальному этажу
        "max_price": 99_000_000  # Ограничение по максимальной цене
    }
)

# Создаём Excel-файл
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Квартиры"

# Записываем заголовки (берём ключи первого объявления)
ws.append(list(data[0].keys()))

# Записываем данные
for flat in data:
    ws.append(list(flat.values()))

# Сохраняем в файл
wb.save("flats.xlsx")

print(f"✅ Сохранено {len(data)} объявлений в flats.xlsx")
