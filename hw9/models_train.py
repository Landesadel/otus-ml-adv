import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import h2o
from h2o.automl import H2OAutoML

train = pd.read_csv('./datasets/train.csv')
test = pd.read_csv('./datasets/test.csv')

# Сохраняем ID теста для сабмита
test_id = test['ID']

# Объединяем данные для согласованной обработки
combined = pd.concat([train, test], axis=0)
combined.drop('ID', axis=1, inplace=True)

# Кодируем категориальные признаки
cat_cols = combined.select_dtypes(include=['object']).columns.tolist()

le = LabelEncoder()
for col in cat_cols:
    combined[col] = le.fit_transform(combined[col].astype(str))

# Разделяем обратно на train/test
train_proc = combined[~combined['y'].isna()]
test_proc = combined[combined['y'].isna()]

# Выделяем фичи и таргет
X = train_proc.drop('y', axis=1)
y = train_proc['y']
X_test = test_proc.drop('y', axis=1)

lr = LinearRegression()
lr_scores = cross_val_score(
  lr,
  X,
  y,
  scoring='neg_root_mean_squared_error',
  cv=5
)
print(f'Linear Regression RMSE: {-lr_scores.mean():.2f}')

# Случайный лес
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf_scores = cross_val_score(
  rf,
  X,
  y,
  scoring='neg_root_mean_squared_error',
  cv=5
)
print(f'Random Forest RMSE: {-rf_scores.mean():.2f}')

# Градиентный бустинг
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_scores = cross_val_score(
  gb,
  X,
  y,
  scoring='neg_root_mean_squared_error',
  cv=5
)
print(f'Gradient Boosting RMSE: {-gb_scores.mean():.2f}')

h2o.init()

# Конвертация в H2O Frame
h2o_train = h2o.H2OFrame(train_proc)
h2o_test = h2o.H2OFrame(X_test)

# Определяем таргет и фичи
target = 'y'
predictors = h2o_train.columns
predictors.remove(target)

# Запуск AutoML (10 минут)
aml = H2OAutoML(
    max_runtime_secs=600,
    seed=42,
    stopping_metric="RMSE",
    sort_metric="RMSE"
)
aml.train(x=predictors, y=target, training_frame=h2o_train)

# Результаты
print("\nLeaderboard:")
print(aml.leaderboard.head())

# Метрики лучшей модели
best_model = aml.leader
print("\nBest Model Performance:")
print(best_model.model_performance())

# Кросс-валидационные метрики
if best_model is not None:
    cv_metrics = best_model.cross_validation_metrics_summary()
    print("\nCross-Validation Metrics:")
    print(cv_metrics.as_data_frame())
