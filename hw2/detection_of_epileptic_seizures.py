import pandas as pd
import numpy as np
import pywt
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


data = pd.read_csv(
    "https://raw.githubusercontent.com/BakerWade/Epileptic-Seizure-recognition/refs/heads/main/Epileptic%20Seizure%20recognition/data.csv",
    sep=","
)

# Unnamed: 0: идентификатор(скорей всего не нужный для анализа).
# X1 - X178: показатели ЭЭГ за 1 секунду(178 измерений).
# y: целевая переменная с классами от 1 до 5.
print(data.head())

# удаление строк с пропусками
if bool(data.isnull().sum().sum()):
    data.dropna(inplace=True)

# удаляем дубликаты:
data.drop_duplicates(inplace=True)

# все признаки числовые, кроме Unnamed: 0 (object) - строковый идентификатор
print(data.dtypes)

data.drop(columns=['Unnamed: 0'], inplace=True)

# Преобразование целевой переменной
data['y'] = data['y'].apply(lambda x: 1 if x == 1 else 0)

print(data['y'].value_counts())
# y
# 0 9200
# 1 2300

X = data.drop('y', axis=1)
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# общая выборка
# y
# 0 7360
# 1 1840

# тестовая выборка:
# y
# 0 1840
# 1 460

# масштабируем данные
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Логистическая регрессия
logreg = LogisticRegression(random_state=42, max_iter=1000)

logreg.fit(X_train_scaled, y_train)

y_pred_logreg = logreg.predict(X_test_scaled)

# Оценка качества модели
print("Качество модели Логистической регрессии на сырых данных:")
print(classification_report(y_test, y_pred_logreg))

#               precision    recall  f1-score   support
#            0       0.81      1.00      0.90      1840
#            1       0.95      0.08      0.15       460
#     accuracy                           0.82      2300
#    macro avg       0.88      0.54      0.52      2300
# weighted avg       0.84      0.82      0.75      2300

# случайный лес
rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Оценка качества модели
print("Качество модели Случайного леса на сырых данных:")
print(classification_report(y_test, y_pred_rf))

#               precision    recall  f1-score   support
#            0       0.98      0.99      0.98      1840
#            1       0.95      0.93      0.94       460
#     accuracy                           0.98      2300
#    macro avg       0.97      0.96      0.96      2300
# weighted avg       0.98      0.98      0.98      2300


# извлечение признаков из сигналов (FFT и Wavelet)
# FTT
def compute_fft(row):
    fft_vals = np.fft.fft(row)
    fft_abs = np.abs(fft_vals)
    return fft_abs[:len(fft_vals)//2]  # только положительные частоты


# применяем FFT к обучающей и тестовой выборкам
X_train_fft = X_train.apply(compute_fft, axis=1, result_type='expand')
X_test_fft = X_test.apply(compute_fft, axis=1, result_type='expand')

# Масштабирование FFT признаков
scaler_fft = StandardScaler()

X_train_fft_scaled = scaler_fft.fit_transform(X_train_fft)
X_test_fft_scaled = scaler_fft.transform(X_test_fft)

# Логистическая регрессия
logreg_fft = LogisticRegression(random_state=42, max_iter=1000)

logreg_fft.fit(X_train_fft_scaled, y_train)
y_pred_logreg_fft = logreg_fft.predict(X_test_fft_scaled)

# Оценка качества модели
print("Качество Логистической регрессии с FFT признаками:")
print(classification_report(y_test, y_pred_logreg_fft))

#               precision    recall  f1-score   support
#            0       0.98      0.99      0.99      1840
#            1       0.96      0.93      0.95       460
#     accuracy                           0.98      2300
#    macro avg       0.97      0.96      0.97      2300
# weighted avg       0.98      0.98      0.98      2300

# случайный лес
rf_fft = RandomForestClassifier(random_state=42)

rf_fft.fit(X_train_fft, y_train)
y_pred_rf_fft = rf_fft.predict(X_test_fft)

# Оценка качества модели
print("Качество Случайного леса с FFT признаками:")
print(classification_report(y_test, y_pred_rf_fft))

#               precision    recall  f1-score   support
#            0       0.99      0.99      0.99      1840
#            1       0.95      0.96      0.95       460
#     accuracy                           0.98      2300
#    macro avg       0.97      0.97      0.97      2300
# weighted avg       0.98      0.98      0.98      2300


# вычисление вейвлет-признаков
def compute_wavelet(row):
    coeffs = pywt.wavedec(row, 'db4', level=4)
    features = np.concatenate(coeffs)
    return features


# Применение вейвлет-преобразования
X_train_wavelet = X_train.apply(compute_wavelet, axis=1, result_type='expand')
X_test_wavelet = X_test.apply(compute_wavelet, axis=1, result_type='expand')

# # масштабирование вейвлет-признаков
scaler_wavelet = StandardScaler()

X_train_wavelet_scaled = scaler_wavelet.fit_transform(X_train_wavelet)
X_test_wavelet_scaled = scaler_wavelet.transform(X_test_wavelet)

# логистическая регрессия
logreg_wavelet = LogisticRegression(random_state=42, max_iter=1000)

logreg_wavelet.fit(X_train_wavelet_scaled, y_train)
y_pred_logreg_wavelet = logreg_wavelet.predict(X_test_wavelet_scaled)

# Оценка качества модели
print("Качество Логистической регрессии с вейвлет-признаками:")
print(classification_report(y_test, y_pred_logreg_wavelet))

#               precision    recall  f1-score   support
#            0       0.82      1.00      0.90      1840
#            1       0.93      0.11      0.19       460
#     accuracy                           0.82      2300
#    macro avg       0.87      0.55      0.55      2300
# weighted avg       0.84      0.82      0.76      2300

# случайный лес
rf_wavelet = RandomForestClassifier(random_state=42)

rf_wavelet.fit(X_train_wavelet, y_train)
y_pred_rf_wavelet = rf_wavelet.predict(X_test_wavelet)

# Оценка качества модели
print("Качество Случайного леса с вейвлет-признаками:")
print(classification_report(y_test, y_pred_rf_wavelet))

#               precision    recall  f1-score   support
#            0       0.98      0.99      0.98      1840
#            1       0.94      0.93      0.93       460
#     accuracy                           0.97      2300
#    macro avg       0.96      0.96      0.96      2300
# weighted avg       0.97      0.97      0.97      2300


# извлечение признаков (tsfresh)
# создание индекса для tsfresh
X_train_selected = X_train.reset_index().rename(columns={'index': 'id'})
X_test_selected = X_test.reset_index().rename(columns={'index': 'id'})

# преобразование данных в длинный формат
df_train_long = pd.melt(X_train_selected, id_vars=['id'], var_name='time', value_name='value')
df_train_long['time'] = df_train_long['time'].str.extract('X(\d+)').astype(int)

df_test_long = pd.melt(X_test_selected, id_vars=['id'], var_name='time', value_name='value')
df_test_long['time'] = df_test_long['time'].str.extract('X(\d+)').astype(int)

# преобразование в секунды (если 178 измерений за 1 секунду)
df_train_long['time'] = df_train_long['time'] / 178.0
df_test_long['time'] = df_test_long['time'] / 178.0

fc_parameters = EfficientFCParameters().data

# извлечение признаков
extracted_features_train = extract_features(
    df_train_long,
    column_id='id',
    column_sort='time',
    default_fc_parameters=fc_parameters,
    impute_function=impute,
    n_jobs=6
)

extracted_features_test = extract_features(
    df_test_long,
    column_id='id',
    column_sort='time',
    default_fc_parameters=fc_parameters,
    impute_function=impute,
    n_jobs=6
)

# отбор значимых признаков
X_filtered = select_features(extracted_features_train, y_train)
X_test_filtered = extracted_features_test[X_filtered.columns]

print(f"Количество признаков после отбора: {X_filtered.shape[1]}")  # 353
print(f"Количество признаков после отбора: {X_test_filtered.shape[1]}")  # 353

# масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)
X_test_scaled = scaler.transform(X_test_filtered)

# PCA
pca = PCA(n_components=0.95, random_state=35)  # сохраняем 95% дисперсии
X_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Количество признаков после PCA: {X_pca.shape[1]}")  # 134

# логистическая регрессия
pipeline = Pipeline([
    ('smote', SMOTE(random_state=50)),
    ('classifier', LogisticRegression(
        max_iter=1000,
        random_state=47,
        class_weight='balanced',
        solver='saga'
    ))
])

# Определение пространства гиперпараметров
param_dist = {
    'smote__sampling_strategy': [0.3, 0.4, 0.5],  # 0.3
    'classifier__penalty': ['l1', 'l2', 'elasticnet', 'none'],  # l2
}

# стратифицированная кросс-валидация
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=45)

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=50,
    scoring='f1',
    cv=skf,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_pca, y_train)

print("Лучшие параметры найденные RandomizedSearchCV:")
print(random_search.best_params_)
print(f"Лучшая оценка F1: {random_search.best_score_:.4f}")

y_pred_rf_tsfresh = random_search.predict(X_test_pca)

# Оценка модели
print("Качество логистической регрессии с признаками tsfresh после применения SMOTE и RandomizedSearchCV:")
print(classification_report(y_test, y_pred_rf_tsfresh))

# Качество логистической регрессии с признаками tsfresh после применения SMOTE и RandomizedSearchCV:
#               precision    recall  f1-score   support
#            0       0.80      0.49      0.61      1840
#            1       0.20      0.49      0.28       460
#     accuracy                           0.49      2300
#    macro avg       0.50      0.49      0.44      2300
# weighted avg       0.68      0.49      0.54      2300

# случайные леса
pipeline = Pipeline([
    ('smote', SMOTE(random_state=70)),
    ('classifier', RandomForestClassifier(random_state=80, class_weight='balanced'))
])

# определение вариаций гиперпараметров
param_dist = {
    'smote__sampling_strategy': [0.3, 0.4, 0.5],  # 0.5
    'classifier__n_estimators': [500, 1000],  # 1000
    'classifier__max_depth': [10, 20, 40, None],  # 10
    'classifier__min_samples_split': [2, 5, 10, 15],  # 2
    'classifier__min_samples_leaf': [1, 2, 4, 6],  # 6
}

# стратифицированная кросс-валидация
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=45)

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=50,
    scoring='f1',
    cv=skf,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_pca, y_train)

print("Лучшие параметры найденные RandomizedSearchCV:")
print(random_search.best_params_)
print(f"Лучшая оценка F1: {random_search.best_score_:.4f}")

y_pred_rf_tsfresh = random_search.predict(X_test_pca)

# Оценка модели
print("Качество Случайного леса с признаками tsfresh после применения SMOTE и RandomizedSearchCV:")
print(classification_report(y_test, y_pred_rf_tsfresh))

# Качество Случайного леса с признаками tsfresh после применения SMOTE и RandomizedSearchCV:
#               precision    recall  f1-score   support
#            0       0.81      0.78      0.79      1840
#            1       0.22      0.25      0.23       460
#     accuracy                           0.67      2300
#    macro avg       0.51      0.51      0.51      2300
# weighted avg       0.69      0.67      0.68      2300
