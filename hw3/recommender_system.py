import datetime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from collections import defaultdict
from scipy import sparse

from catboost import CatBoostClassifier
import implicit

# загружаем датасеты
interactions = pd.read_csv("datasets/interactions.csv")
items = pd.read_csv("datasets/items.csv")
users = pd.read_csv("datasets/users.csv")

# работаем с данными взаимодействия пользователей и айтемов (фильмы)
interactions.head()
#    user_id  item_id last_watch_dt  total_dur  watched_pct
# 0   176549     9506    2021-05-11       4250         72.0
# 1   699317     1659    2021-05-29       8317        100.0
# 2   656683     7107    2021-05-09         10          0.0
# 3   864613     7638    2021-07-05      14483        100.0
# 4   964868     9506    2021-04-30       6725        100.0

# last_watch_dt дата просмотра
# total_dur длительность просмотра
# watched_pct доля просмотра в %

# Преобразование даты
interactions['last_watch_dt'] = pd.to_datetime(interactions['last_watch_dt']).map(lambda x: x.date())

print(f"min дата в interactions: {interactions['last_watch_dt'].max()}")
print(f"max дата в interactions: {interactions['last_watch_dt'].min()}")

# проверка столбца watched_pct на пустые значения
missing_pct = interactions['watched_pct'].isnull().mean() * 100
print(f"Процент пропусков в watched_pct: {missing_pct:.2f}%")  # 0.2

# удаляем пропуски
interactions.dropna(subset=['watched_pct'], inplace=True)

# фильтрация для исключения случайных просмотров
interactions = interactions[interactions['total_dur'] >= 300]

# работа с данными пользователей
users.head()
#    user_id        age        income sex  kids_flg
# 0   973171  age_25_34  income_60_90   М         1
# 1   962099  age_18_24  income_20_40   М         0
# 2  1047345  age_45_54  income_40_60   Ж         0
# 3   721985  age_45_54  income_20_40   Ж         0
# 4   704055  age_35_44  income_60_90   Ж         0

# age бин по возрасту
# income бин по доходу //имхо: ненужный признак
# sex пол
# kids_flg флаг наличия детей

# Получение уникальных значений столбца - возраст
age_counts = users['age'].value_counts()

print(age_counts)

# проверка столбца age на пустые значения
missing_age = users['age'].isnull().mean() * 100
print(f"Процент пропусков в age: {missing_age:.2f}%")  # 1.68

# удаляем строчки с пропущенным age
users.dropna(subset=['age'], inplace=True)

# применяем one-hot encoding к столбцу 'age'
age_dummies = pd.get_dummies(users['age'])
age_dummies = age_dummies.astype(int)
users = pd.concat([users, age_dummies], axis=1)
users.drop('age', axis=1, inplace=True)

# проверка столбца sex на пустые значения
missing_sex = users['sex'].isnull().mean() * 100
print(f"Процент пропусков в sex: {missing_sex:.2f}%")  # 0.64

# удаляем пропущенные значения
users.dropna(subset=['sex'], inplace=True)

# На всякий проверим уникальность значений, вдруг там модное ОНО
age_counts = users['sex'].value_counts()

print(age_counts)

# переводим колонку sex в бинарный формат, где М: 1 а ж: 0
users['sex'] = users['sex'].apply(lambda x: 0 if x == 'Ж' else 1)

print(users)

# удаляем ненужную колонку с доходом
users.drop('income', axis=1, inplace=True)
print(users.head())

# работа с данными айтемов
print(items.head(2))

#      Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   item_id       15963 non-null  int64
#  1   content_type  15963 non-null  object
#  2   title         15963 non-null  object
#  3   title_orig    11218 non-null  object
#  4   release_year  15865 non-null  float64
#  5   genres        15963 non-null  object
#  6   countries     15926 non-null  object
#  7   for_kids      566 non-null    float64
#  8   age_rating    15961 non-null  float64
#  9   studios       1065 non-null   object
#  10  directors     14454 non-null  object
#  11  actors        13344 non-null  object
#  12  description   15961 non-null  object
#  13  keywords      15540 non-null  object

# заполнение пропусков
string_columns = ['title_orig', 'studios', 'directors', 'actors', 'keywords']
items[string_columns] = items[string_columns].fillna('')

# для признаков с числовыми значениями
items.dropna(subset=['release_year'], inplace=True)
items['for_kids'] = items['for_kids'].fillna(0)  # предположим, что 0 - отсутствие
items['age_rating'] = items['age_rating'].fillna(items['age_rating'].median())

# подсчёт популярности элементов
item_popularity = interactions.groupby('item_id').size().reset_index(name='popularity')
items = items.merge(item_popularity, on='item_id', how='left').fillna(0)

# разбивка жанров на отдельные бинарные признаки
genres = items['genres'].str.get_dummies(sep=', ')
items = pd.concat([items, genres], axis=1)

items.drop('genres', axis=1, inplace=True)

# определим текущую дату как максимальную дату в данных
current_date = interactions['last_watch_dt'].max()

# фильтруем взаимодействия за последние 30 дней
recent_interactions = interactions[interactions['last_watch_dt'] >= (current_date - pd.Timedelta(days=30))]

# подсчёт популярности за всё время и за последние 30 дней
overall_popularity = interactions.groupby('item_id').size().reset_index(name='overall_popularity')
recent_popularity = recent_interactions.groupby('item_id').size().reset_index(name='recent_popularity')

# объединяем с таблицей items
items = items.merge(overall_popularity, on='item_id', how='left').fillna(0)
items = items.merge(recent_popularity, on='item_id', how='left').fillna(0)

# сортируем взаимодействия по дате
interactions = interactions.sort_values('last_watch_dt')

# разделим данные: последние 20% взаимодействий - test, предыдущие 10% - validation, остальные - train
train_size = int(0.7 * len(interactions))
val_size = int(0.1 * len(interactions))

train = interactions.iloc[:train_size]
val = interactions.iloc[train_size:train_size + val_size]
test = interactions.iloc[train_size + val_size:]

print(f"train: {train.head()}")
print(f"val: {val.head()}")
print(f"test: {test.head()}")

# первая модель

# фильтрация валидационной и тестовой выборки только для "теплых" пользователей
val = val[val['user_id'].isin(train['user_id'].unique())]
train = val[val['user_id'].isin(train['user_id'].unique())]

# получение уникальных идентификаторов пользователей и айтемов
users_id = list(np.sort(train.user_id.unique()))
items_train = list(train.item_id.unique())

# извлечение рейтингов и кодирование идентификаторов в числовые индексы
ratings_train = list(train.watched_pct)

# Создание категориальных типов с определёнными категориями
train['user_id'] = train['user_id'].astype('category')
train['user_id'].cat.set_categories(users_id)

train['item_id'] = train['item_id'].astype('category')
train['item_id'].cat.set_categories(items_train)

# Получение кодов уже с учётом порядка
rows_train = train['user_id'].cat.codes
cols_train = train['item_id'].cat.codes

# создание разреженной матрицы взаимодействий
train_sparse = sparse.csr_matrix(
    (ratings_train, (rows_train, cols_train)),
    shape=(len(users_id), len(items_train))
)

# Вычисление разреженности матрицы
matrix_size = train_sparse.shape[0] * train_sparse.shape[1]
num_interactions = len(train_sparse.nonzero()[0])
sparsity = 100 * (1 - (num_interactions / matrix_size))
print(f"разреженность: {sparsity:.2f}%")

als_model = implicit.als.AlternatingLeastSquares(
    factors=50,
    regularization=0.01,
    iterations=50,
    use_gpu=False
)

# обучение модели, используем матрицу в формате CSC
train_csc = train_sparse.tocsc()
als_model.fit(train_csc)

# извлечение векторов пользователей и айтемов
user_vecs = als_model.user_factors
item_vecs = als_model.item_factors


def predict_in_batches(user_vecs, item_vecs, interaction_matrix, id_user, id_item, k=20, batch_size=1000):
    num_users = user_vecs.shape[0]
    preds = []

    for start in tqdm(range(0, num_users, batch_size), desc="Предсказание по каждому батчу"):
        end = min(start + batch_size, num_users)
        batch_user_vecs = user_vecs[start:end]

        # вычисление скорингов для батча пользователей
        scores = np.dot(batch_user_vecs, item_vecs.T)

        # преобразование взаимодействий в плотный формат для фильтрации
        batch_interaction = interaction_matrix[start:end].toarray().astype(bool)

        # фильтрация уже просмотренных айтемов
        scores[batch_interaction] = -np.inf  # ставим низкие значения для уже просмотренных

        # Получение топ-k индексов
        top_k_indices = np.argpartition(scores, -k, axis=1)[:, -k:]
        top_k_scores = np.take_along_axis(scores, top_k_indices, axis=1)
        sorted_top_k_indices = np.argsort(top_k_scores, axis=1)[:, ::-1]
        top_k_sorted_indices = np.take_along_axis(top_k_indices, sorted_top_k_indices, axis=1)

        # Преобразование индексов в реальные item_id
        for i, user_idx in enumerate(range(start, end)):
            user_id = id_user[user_idx]
            item_ids = [id_item[idx] for idx in top_k_sorted_indices[i]]
            preds.append({'user_id': user_id, 'preds': item_ids})

    return pd.DataFrame(preds)


# преобразуем индексы в ID
id_user = dict(zip(range(train_sparse.shape[0]), train.user_id.unique()))
id_item = dict(zip(range(train_sparse.shape[1]), train.item_id.unique()))

# получение рекомендаций
pred_als = predict_in_batches(user_vecs, item_vecs, train_sparse, id_user, id_item, k=20, batch_size=1000)

# объединение с историей пользователей из валидационной выборки
val_user_history = val.groupby('user_id')['item_id'].apply(list).reset_index()
pred_als = val_user_history.merge(pred_als, how='left', on='user_id')

print(pred_als.head())


def recall(df: pd.DataFrame, pred_col='preds', true_col='item_id', k=30) -> float:
    recall_values = []
    for _, row in df.iterrows():
      num_relevant = len(set(row[true_col]) & set(row[pred_col][:k]))
      num_true = len(row[true_col])
      recall_values.append(num_relevant / num_true)
    return np.mean(recall_values)


def precision(df: pd.DataFrame, pred_col='preds', true_col='item_id', k=30) -> float:
    precision_values = []
    for _, row in df.iterrows():
      num_relevant = len(set(row[true_col]) & set(row[pred_col][:k]))
      num_true = min(k, len(row[true_col]))
      precision_values.append(num_relevant / num_true)
    return np.mean(precision_values)


def mrr(df: pd.DataFrame, pred_col='preds', true_col='item_id', k=30) -> float:
    mrr_values = []
    for _, row in df.iterrows():
      intersection = set(row[true_col]) & set(row[pred_col][:k])
      user_mrr = 0
      if len(intersection) > 0:
          for item in intersection:
              user_mrr = max(user_mrr, 1 / (row[pred_col].index(item) + 1))
      mrr_values.append(user_mrr)
    return np.mean(mrr_values)


print(recall(pred_als))
print(precision(pred_als))
print(mrr(pred_als))

# результаты не самые лучшие, возможно из-за сильной разряженности матрицы
# 0.0019935785627680502
# 0.0019941783846437527
# 0.000986762558459491


# Вторая модель
# положительные примеры
positive = train[['user_id', 'item_id']].copy()
positive['interaction'] = 1

# выберем N негативных товаров
N_neg = 5

# множество всех товаров
all_items = set(train['item_id'].unique())

# истории взаимодействий пользователей
user_history = train.groupby('user_id')['item_id'].apply(set).to_dict()

negative = []
for user in tqdm(user_history.keys(), desc="Генерация негативных примеров"):
    neg_items = all_items - user_history[user]
    neg_samples = np.random.choice(list(neg_items), size=N_neg, replace=False)
    for item in neg_samples:
        negative.append({'user_id': user, 'item_id': item, 'interaction': 0})

negative = pd.DataFrame(negative)

# объединяем положительные и негативные примеры
data = pd.concat([positive, negative], ignore_index=True)


# функция для получения векторов
def get_user_item_vectors(df, id_user, id_item):
    user_indices = df['user_id'].map(id_user).values
    item_indices = df['item_id'].map(id_item).values

    user_features = user_vecs[user_indices]
    item_features = item_vecs[item_indices]

    # Объединяем векторы пользователя и товара
    features = np.hstack([user_features, item_features])

    return features


# делаем словари ID к индексам
id_user_map = dict(zip(train.user_id.unique(), range(len(train.user_id.unique()))))
id_item_map = dict(zip(train.item_id.unique(), range(len(train.item_id.unique()))))

# проверка, что все user_id и item_id имеют соответствующие индексы
data = data[data['user_id'].isin(id_user_map)]
data = data[data['item_id'].isin(id_item_map)]

# извлекаем признаки
features = get_user_item_vectors(data, id_user_map, id_item_map)

# создаем DataFrame для CatBoost
feature_columns = (
    [f'user_vec_{i}' for i in range(user_vecs.shape[1])] +
    [f'item_vec_{i}' for i in range(item_vecs.shape[1])]
)

features_df = pd.DataFrame(features, columns=feature_columns)
features_df['interaction'] = data['interaction'].values

X = features_df[feature_columns]
y = features_df['interaction']

X_train, X_val, y_train, y_val, data_train, data_val = train_test_split(
    X, y, data, test_size=0.2, random_state=42, stratify=y
)

cat_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    eval_metric='AUC',
    random_seed=42,
    verbose=100,
    use_best_model=True
)

cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50
)

# 0:      test: 0.6337078 best: 0.6337078 (0)     total: 181ms    remaining: 3m 1s
# 100:    test: 0.8367704 best: 0.8367704 (100)   total: 13.6s    remaining: 2m
# 200:    test: 0.8661382 best: 0.8661382 (200)   total: 32.9s    remaining: 2m 10s
# 300:    test: 0.8796132 best: 0.8796132 (300)   total: 42.8s    remaining: 1m 39s
# 400:    test: 0.8875606 best: 0.8875606 (400)   total: 52.6s    remaining: 1m 18s
# 500:    test: 0.8929951 best: 0.8929951 (500)   total: 1m 3s    remaining: 1m 2s
# 600:    test: 0.8964186 best: 0.8964186 (600)   total: 1m 13s   remaining: 48.6s
# 700:    test: 0.8993352 best: 0.8993352 (700)   total: 1m 24s   remaining: 36.2s
# 800:    test: 0.9015132 best: 0.9015132 (800)   total: 1m 35s   remaining: 23.6s
# 900:    test: 0.9032763 best: 0.9032763 (900)   total: 1m 46s   remaining: 11.7s
# 999:    test: 0.9045552 best: 0.9045552 (999)   total: 1m 58s   remaining: 0us

# bestTest = 0.9045552417
# bestIteration = 999

# Оценка модели, метрики
y_pred_proba = cat_model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred_proba)
print(f'AUC на валидационной выборке: {auc:.4f}')  # 90%

y_pred = (y_pred_proba >= 0.5).astype(int)
acc = accuracy_score(y_val, y_pred)
print(f'Accuracy на валидационной выборке: {acc:.4f}')  # 84%

# предположения модели для валидационной выборки
validation_df = data_val.copy()
validation_df['pred_proba'] = y_pred_proba

# группируем данные по пользователям и сортируем предсказания
grouped = validation_df.groupby('user_id')

# списки для формирования нового DataFrame
users = []
preds = []
true_items = []

K = 20  # количество рекомендаций

for user, group in tqdm(grouped, desc="Подготовка данных для метрик"):
    # сортируем элементы по убыванию предсказанных вероятностей
    group_sorted = group.sort_values(by='pred_proba', ascending=False)
    top_k = group_sorted.head(K)['item_id'].tolist()

    # истинные элементы пользователя
    true_user_items = list(user_history.get(user, []))

    users.append(user)
    preds.append(top_k)
    true_items.append(true_user_items)

metrics_df = pd.DataFrame({
    'user_id': users,
    'preds': preds,
    'item_id': true_items
})

# K=20 для метрик
K_METRIC = 20

precision_at_k = precision(metrics_df, pred_col='preds', true_col='item_id', k=K_METRIC)
recall_at_k = recall(metrics_df, pred_col='preds', true_col='item_id', k=K_METRIC)
mrr_at_k = mrr(metrics_df, pred_col='preds', true_col='item_id', k=K_METRIC)

print(f'Precision@{K_METRIC}: {precision_at_k:.4f}')
print(f'Recall@{K_METRIC}: {recall_at_k:.4f}')
print(f'MRR@{K_METRIC}: {mrr_at_k:.4f}')
