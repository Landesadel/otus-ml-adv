# 🩺 Heart Disease Prediction API

Прогнозирование сердечно-сосудистых заболеваний с использованием нейронной сети на TensorFlow. Модель развернута как REST API с помощью FastAPI и Docker.

---

## 🚀 Особенности
- Полный пайплайн EDA (анализ данных, обработка выбросов, визуализации)
- Нейронная сеть с регуляризацией (Dropout)
- Автоматическая нормализация данных
- Мониторинг метрик обучения (Accuracy, Precision, Recall)
- Готовый Docker-образ для production

## 📋 Требования
- Docker >= 24.0
- Python 3.12
- Git
- Скачать датасет с https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/ и сохранить по пути **./datasets/heart.csv**

## ⚡ Быстрый старт

### 1. Клонировать репозиторий
```bash
git clone https://github.com/yourusername/heart-api.git
cd heart-api
```
### 2. Установить зависимости
```bash
pip install -r requirements.txt
```
### 3. Запустить обучение модели
```bash
python train_tf_with_eda.py
```

### 4. Собрать Docker-образ
```bash
docker build -t heart-api .
```

### 5. Запустить контейнер
```bash
docker run -p 8000:8000 heart-api
```

## Примеры запросов
**GET-запрос (проверка работы API)**
```bash
curl http://localhost:8000
```

**POST-запрос (прогноз)**
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "age": 57,
    "sex": 1,
    "cp": 0,
    "trestbps": 130,
    "chol": 236,
    "fbs": 0,
    "restecg": 1,
    "thalach": 174,
    "exang": 0,
    "oldpeak": 0.0,
    "slope": 1,
    "ca": 1,
    "thal": 2
}'
```

**Тестирование через Python**
Файл curl_test.py содержит готовый пример для тестирования:
```bash
python curl_test.py
```

## 🧠 Обучение модели
Модель обучается с полным EDA пайплайном:

- Анализ распределений
- Обработка выбросов
- Визуализация корреляций
- Автоматическое сохранение артефактов

## Артефакты обучения:

- models/heart_model.keras - сохраненная модель
- eda_plots/* - графики анализа данных
- eda_plots/training_history.png - динамика обучения

## 🐳 Docker-развертывание
Конфигурация оптимизирована для Python 3.12:

```dockerfile
FROM python:3.12-slim
```

Проверка версий в контейнере:

```bash
docker exec -it <container_id> python --version
docker exec -it <container_id> pip list
```
