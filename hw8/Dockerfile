FROM python:3.12-slim
LABEL authors="landesadel"

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./app
COPY ./models/heart_model.keras .app/models/heart_model.keras
COPY scaler_mean.npy scaler_scale.npy ./

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]