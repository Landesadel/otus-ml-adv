import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# Загрузка данных
df = pd.read_csv('./datasets/heart.csv')

# 1. Общая информация
print("\n[1] Первые 5 строк:")
print(df.head())

print("\n[2] Информация о данных:")
print(df.info())

# 2. Проверка пропусков
missing_values = df.isnull().sum()
print("\n[3] Пропущенные значения:")
print(missing_values)

# 3. Анализ целевой переменной
plt.figure(figsize=(8,6))
sns.countplot(x='target', data=df)
plt.title('Распределение целевой переменной')
plt.savefig('eda_plots/target_distribution.png')
plt.close()

print("\n[4] Распределение классов:")
print(df['target'].value_counts(normalize=True))

# 4. Поиск дубликатов
duplicates = df.duplicated().sum()
print(f"\n[5] Найдено дубликатов: {duplicates}")

# Удаление дубликатов
if duplicates > 0:
    print("Удаляем дубликаты...")
    df = df.drop_duplicates()

# 5. Анализ выбросов
plt.figure(figsize=(12, 8))
df.boxplot()
plt.xticks(rotation=45)
plt.title('Распределение признаков (ящики с усами)')
plt.savefig('eda_plots/boxplots.png', bbox_inches='tight')
plt.close()

print("\n[6] Описательная статистика:")
print(df.describe().T)

# 6. Корреляционный анализ
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Матрица корреляций')
plt.savefig('eda_plots/correlation_matrix.png')
plt.close()

# Разделение на признаки и целевую переменную
X = df.drop('target', axis=1)
y = df['target'].values

# Нормализация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Сохранение параметров нормализации
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('./models/best_model.keras', save_best_only=True)
]

# Обучение
history = model.fit(
    X_scaled, y,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=2,
    callbacks=callbacks
)

plt.figure(figsize=(12,6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Точность')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Потери')
plt.legend()

plt.savefig('eda_plots/training_history.png')
plt.close()

# Сохранение финальной модели
model.save('./models/heart_model.keras')

print("\nОбучение завершено! Результаты EDA сохранены в папке eda_plots")
