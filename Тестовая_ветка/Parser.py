import cianparser
import openpyxl
import requests
from bs4 import BeautifulSoup

# Функция для получения времени до метро
def get_metro_time(url):
    try:
        # Запрашиваем страницу объявления
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Ищем элемент с временем до метро
        metro_time_tag = soup.find('span', class_='a10a3f92e9--underground_time--YvrcI')  # Класс по вашему примеру
        if metro_time_tag:
            # Извлекаем текст и парсим количество минут
            metro_time_text = metro_time_tag.text.strip()  # Например: "8 мин."
            try:
                metro_time = int(metro_time_text.split()[0])  # Извлекаем число до слова "мин"
                return metro_time
            except ValueError:
                return None  # Если не удалось преобразовать в число
        return None  # Если не нашли информации о времени
    except requests.exceptions.RequestException:
        return None  # Если произошла ошибка при запросе страницы

def get_build_year(url):
    try:
        # Запрашиваем страницу объявления
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Ищем все элементы <div> с нужным классом
        build_year_label = soup.find('p', text='Год постройки')  # Ищем по метке "Год постройки"
        
        if build_year_label:
            # Находим следующий тег <p> с годом
            build_year_tag = build_year_label.find_next('p')
            if build_year_tag:
                return build_year_tag.text.strip()  # Возвращаем текст (например, "1979")
        
        return None  # Если не нашли информации о годе постройки
    except requests.exceptions.RequestException:
        return None  # Если произошла ошибка при запросе страницы

# Создаём парсер для Москвы
moscow_parser = cianparser.CianParser(location="Москва")

# Собираем данные (~500 записей) с фильтрами для этажей квартиры и здания, а также с увеличенным лимитом по цене
data = moscow_parser.get_flats(
    deal_type="sale",
    rooms=(1, 2, 3, 4),
    with_saving_csv=True,
    additional_settings={
        "start_page": 1,
        "end_page": 1,
        "min_area": 30,  # Студии слишком разные, лучше с 30 м²
        "max_area": 150,
        "min_floor": 2,
        "max_floor": 30,  # Ограничение по максимальному этажу квартиры
        "max_total_floor": 40,  # Ограничение по максимальному этажу здания
        "min_price": 9_000_000,
        "max_price": 57_000_000,  # Ограничение по максимальной цене (200 млн)
        "only_flat": True,
        "min_house_year": 1975,  # Исключаем совсем старые хрущёвки
        "max_house_year": 2019,
        "metro_foot_minute": 15,
        "min_balconies": 1,
        "house_material_type": [1, 2, 3],  # Кирпич, панель, монолит
        "object_type": "secondary"  # Вторичка
    }
)

# Создаём Excel-файл
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Квартиры"

# Записываем заголовки (берём ключи первого объявления)
ws.append(["url", "floor", "floors_count", "rooms_count", "total_meters", "price", "district", "underground", "цена за метр", "время до метро", "год постройки"])

# Записываем данные
total_flats = len(data)
for idx, flat in enumerate(data):
    # Выводим прогресс
    progress = (idx + 1) / total_flats * 100
    print(f"Парсим: {progress:.2f}% завершено", end="\r")

    # Проверяем наличие метро и района
    if flat.get("underground") and flat.get("district"):
        # Расчёт цены за квадратный метр
        price_per_meter = flat["price"] / flat["total_meters"] if flat["total_meters"] != 0 else 0
        
        # Получаем время до метро
        metro_time = get_metro_time(flat["url"])
        
        # Получаем год постройки
        build_year = get_build_year(flat["url"])
        
        # Добавляем данные в Excel
        ws.append([
            flat["url"], 
            flat["floor"], 
            flat["floors_count"], 
            flat["rooms_count"], 
            flat["total_meters"], 
            flat["price"], 
            flat["district"],  
            flat["underground"],
            price_per_meter,  # Цена за метр
            metro_time,  # Время до метро
            build_year  # Год постройки
        ])

# Сохраняем в файл
wb.save("flats.xlsx")

print(f"✅ Сохранено {len(data)} объявлений в flats.xlsx")
