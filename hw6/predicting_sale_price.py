import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

print('Загрузка данных')
# загружаем датасеты
test_data = pd.read_csv('datasets/test.csv')
train_data = pd.read_csv('datasets/train.csv')
sample_sub_data = pd.read_csv('datasets/sample_submission.csv')


# функция для обработки пропусков
def fill_missing(data):
    # Определение числовых и категориальных колонок
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Заполнение числовых пропусков медианой
    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

    # Заполнение категориальных пропусков модой
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

    return data


print('Подготовка данных')
# подготовка данных
train_data = fill_missing(train_data)
test_data = fill_missing(test_data)

# отделение целевой переменной
y_train = train_data['SalePrice']
train_data = train_data.drop(['SalePrice'], axis=1)

# One-Hot Encoding для всех данных вместе, чтобы избежать несоответствия колонок
combined = pd.concat([train_data, test_data], sort=False).reset_index(drop=True)
combined = pd.get_dummies(combined, drop_first=True)

# разделение обратно на обучающую и тестовую выборки
train_processed = combined.iloc[:len(train_data)]
test_processed = combined.iloc[len(train_data):]

# выравнивание колонок после One-Hot Encoding
train_processed, test_processed = train_processed.align(test_processed, join='left', axis=1, fill_value=0)

train_processed['SalePrice'] = y_train.values


# Функция для удаления выбросов
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


# удаления выбросов ко всем числовым колонкам, кроме целевой
numerical_cols = train_processed.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    if col != 'SalePrice':
        train_processed = remove_outliers(train_processed, col)

# логарифмирование целевой переменной
train_processed['SalePrice'] = np.log(train_processed['SalePrice'])

# подготовка признаков и целевой переменной
X_train = train_processed.drop(['Id', 'SalePrice'], axis=1).values
y_train = train_processed['SalePrice'].values
X_test = test_processed.drop(['Id'], axis=1).values

# масштабирование признаков и целевой переменной
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

print('Инициализация модели')
model = lgb.LGBMRegressor(random_state=42)

param_grid = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)

print('Обучение модели')
grid_search.fit(X_train_scaled, y_train_scaled)

# вывод лучших параметров и RMSE
print(f'Вывод лучших параметров: {grid_search.best_params_}')
# {'colsample_bytree': 0.8, 'learning_rate': 0.1, 'n_estimators': 100, 'num_leaves': 31, 'subsample': 0.8}
best_rmse = np.sqrt(-grid_search.best_score_)
print(f'Лучший результат RMSE: {best_rmse}')

# извлечение лучшей модели из GridSearchCV
best_model = grid_search.best_estimator_

print('Проверка лучшей модели, на тестовых данных')
y_pred_scaled = best_model.predict(X_test_scaled)

# обратное масштабирование предсказаний
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# обратное логарифмирование предсказаний
y_pred = np.exp(y_pred)

# загрузка истинных значений из sample_sub_data
true_prices = sample_sub_data['SalePrice'].values

# логарифмирование истинных значений и предсказаний
log_true_prices = np.log(true_prices)
log_pred_prices = np.log(y_pred)

# расчет RMSE между логарифмами истинных значений и предсказаний
rmse = np.sqrt(mean_squared_error(log_true_prices, log_pred_prices))
print(f'RMSE: {rmse}')
