from fastapi import FastAPI
import numpy as np
import tensorflow as tf

from schemas.heart_data import HeartData

app = FastAPI()
model = tf.keras.models.load_model('./models/heart_model.keras')
scaler_mean = np.load('scaler_mean.npy')
scaler_scale = np.load('scaler_scale.npy')


@app.post("/predict")
def predict(data: HeartData):
    # Преобразование в массив и нормализация
    raw_data = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])

    scaled_data = (raw_data - scaler_mean) / scaler_scale

    # Предсказание
    prediction = model.predict(scaled_data)
    return {"prediction": float(prediction[0][0]), "class": int(prediction[0][0] > 0.5)}

