import cianparser
import openpyxl

# Создаём парсер для Москвы
moscow_parser = cianparser.CianParser(location="Москва")

# Собираем данные (~500 записей) с фильтрами для этажей квартиры и здания, а также с увеличенным лимитом по цене
data = moscow_parser.get_flats(
    deal_type="sale",
    rooms=(1, 2, 3, 4, 5),
    with_saving_csv=True,
    additional_settings={
        "start_page": 1,
        "end_page": 20,
        "max_floor": 45,  # Ограничение по максимальному этажу квартиры
        "max_total_floor": 50,  # Ограничение по максимальному этажу здания
        "max_price": 200_000_000,  # Ограничение по максимальной цене (200 млн)
        "only_flat": True,
        "max_house_year": 2022,  # Ограничение по году постройки
        "object_type": "secondary"  # Вторичка
    }
)

# Создаём Excel-файл
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Квартиры"

# Записываем заголовки (берём ключи первого объявления)
ws.append(["url", "floor", "floors_count", "rooms_count", "total_meters", "price", "district", "underground", "residential_complex"])

# Записываем данные
for flat in data:
    # Проверяем наличие метро и района
    if flat.get("underground") and flat.get("district"):
        ws.append([
            flat["url"], 
            flat["floor"], 
            flat["floors_count"], 
            flat["rooms_count"], 
            flat["total_meters"], 
            flat["price"], 
            flat["district"],  
            flat["underground"], 
            flat.get("residential_complex", "")
        ])

# Сохраняем в файл
wb.save("flats.xlsx")

print(f"✅ Сохранено {len(data)} объявлений в flats.xlsx")