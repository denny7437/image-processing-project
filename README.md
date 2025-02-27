# Проект по обработке изображений

Проект демонстрирует различные методы обработки изображений с использованием Python, OpenCV и других библиотек компьютерного зрения.

## Описание

Проект включает в себя реализацию следующих задач обработки изображений:

1. Загрузка и отображение изображений
2. Преобразование цветовых пространств (RGB, Grayscale, HSV)
3. Применение различных фильтров:
   - Гауссово сглаживание
   - Медианная фильтрация
   - Фильтр Лапласа
4. Определение краев и углов:
   - Оператор Собеля
   - Детектор краев Кэнни
   - Детектор углов Харриса
5. Морфологические операции:
   - Бинаризация
   - Эрозия
   - Дилатация

## Структура проекта 

## Требования

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- Pillow
- scikit-image

## Установка

1. Клонируйте репозиторий:

## Использование

1. Поместите изображение для обработки в директорию `images/`

2. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # для Linux/macOS
venv\Scripts\activate     # для Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Запустите скрипт:
```bash
python main.py
```

5. Результаты обработки будут сохранены в директории `results/` в соответствующих поддиректориях для каждого задания.

## Результаты

Каждое задание создает свой набор результатов:

- **Задание 1**: Исходное изображение
- **Задание 2**: Цветовые пространства и гистограммы
- **Задание 3**: Результаты применения различных фильтров
- **Задание 4**: Результаты определения краев и углов
- **Задание 5**: Результаты морфологических операций

## Лицензия

MIT

## Автор

Нагаев Денис

## Благодарности

- OpenCV team
- NumPy developers
- Matplotlib team 