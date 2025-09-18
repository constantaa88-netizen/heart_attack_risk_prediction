Heart Attack Risk Prediction Project
Описание проекта.
  Проект для предсказания риска сердечного приступа на основе медицинских данных пациента. Включает в себя полный цикл обработки данных: исследование, предобработку, обучение модели машинного обучения и развертывание REST API с использованием FastAPI.

Структура проекта:

heart_attack_risk_prediction/
├── data/                    # Данные
│   ├── heart_train.csv      # Обучающая выборка
│   └── test.csv            # Тестовая выборка
├── notebooks/              # Jupyter ноутбуки
│   └── eda.ipynb          # Исследовательский анализ данных
├── src/                    # Исходный код
│   ├── __init__.py
│   ├── preprocessor.py    # Класс для предобработки данных
│   └── model.py          # Класс модели машинного обучения
├── scripts/               # Скрипты
│   ├── train_model.py    # Обучение модели
│   └── predict.py        # Предсказание на тестовых данных
├── app/                  # FastAPI приложение
│   └── main.py          # Основной файл API
├── models/               # Сохраненные модели
├── predictions/          # Результаты предсказаний
├── requirements.txt      # Зависимости Python
└── README.md            # Этот файл

Установка и запуск:


1. Клонирование репозитория:
git clone <URL вашего репозитория>
cd heart_attack_risk_prediction

2. Создание и активация окружения (рекомендуется):
conda create -n heart-attack-env python=3.9
conda activate heart-attack-env

3. Установка зависимостей:
pip install -r requirements.txt

4. Обучение модели:
python scripts/train_model.py

5. Создание предсказаний на тестовых данных:
python scripts/predict.py

6. Запуск FastAPI сервера:
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 (После запуска сервер будет доступен по адресу: http://localhost:8000)

Использование API
Документация API
После запуска сервера доступны следующие URL:

Основная страница: http://localhost:8000

Интерактивная документация (Swagger UI): http://localhost:8000/docs

Альтернативная документация (ReDoc): http://localhost:8000/redoc

Эндпоинты API:

1. Проверка здоровья приложения:
curl -X GET http://localhost:8000/health ответ ({"status":"healthy","model_loaded":true})

2. Предсказание для одного пациента:
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"data\": {\"Age\": 45, \"Cholesterol\": 200, \"Heart rate\": 75, \"Diabetes\": 0, \"Family History\": 1, \"Smoking\": 1, \"Obesity\": 0, \"Alcohol Consumption\": 1, \"Exercise Hours Per Week\": 3, \"Diet\": 1, \"Previous Heart Problems\": 0, \"Medication Use\": 0, \"Stress Level\": 5, \"Sedentary Hours Per Day\": 8, \"Income\": 50000, \"BMI\": 25, \"Triglycerides\": 150, \"Physical Activity Days Per Week\": 3, \"Sleep Hours Per Day\": 7, \"Blood sugar\": 100, \"C-reactive protein\": 0.5, \"Troponin\": 0.01, \"Gender\": \"Male\", \"Systolic blood pressure\": 120, \"Diastolic blood pressure\": 80}}"
Ответ: ({
  "prediction": 0,
  "probabilities": [0.85, 0.15]
})

3. Пакетная обработка данных из CSV файла:
curl -X POST "http://localhost:8000/predict_batch?file_path=data/test.csv"
Ответ: ({
  "predictions": [
    {"id": 1, "prediction": 0},
    {"id": 2, "prediction": 1},
    ...
  ]
})

Формат данных
Входные данные
Модель ожидает данные со следующими признаками:

Антропометрические параметры: Age, Cholesterol, Heart rate, BMI

Привычки: Smoking, Obesity, Alcohol Consumption, Exercise Hours Per Week, Sedentary Hours Per Day, Physical Activity Days Per Week, Sleep Hours Per Day

Давление: Systolic blood pressure, Diastolic blood pressure

Хронические заболевания: Diabetes, Family History, Previous Heart Problems

Биохимия крови: Triglycerides, Blood sugar, C-reactive protein, Troponin

Демографические данные: Gender, Income

Выходные данные
Для единичного предсказания: вероятность и бинарное предсказание (0 - низкий риск, 1 - высокий риск)

Для пакетной обработки: CSV файл с колонками id и prediction

