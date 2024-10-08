import pandas as pd
import networkx as nx
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# список ребер графа
edges = pd.read_csv(
    "https://raw.githubusercontent.com/a-milenkin/Otus_HW_protein_expression/main/edges.csv",
    sep=","
)

# подгрузим тренировочную выборку
train = pd.read_csv(
    "https://raw.githubusercontent.com/a-milenkin/Otus_HW_protein_expression/main/train.csv",
    sep=","
)

# подгрузим отложенную выборку для валидации
test = pd.read_csv(
    "https://raw.githubusercontent.com/a-milenkin/Otus_HW_protein_expression/main/test.csv",
    sep=","
)


# функция для вычисления графовых признаков
def compute_graph_features(graph):
    features = pd.DataFrame()
    degree_dict = dict(graph.degree())  # степень узла
    clustering_dict = nx.clustering(graph)  # кластерный коэффициент
    #  вычисляем центральность между узлами в графе
    betweenness_dict = nx.betweenness_centrality(
        graph,
        k=1000,  # кол-во случайных пар узлов (меняемый параметр)
        normalized=True,
        endpoints=True
    )
    pagerank_dict = nx.pagerank(graph)  # показатель PageRank

    # собираем датафрейм
    features['node'] = list(graph.nodes())
    features['degree'] = [degree_dict[node] for node in features['node']]
    features['clustering'] = [clustering_dict[node] for node in features['node']]
    features['betweenness'] = [betweenness_dict[node] for node in features['node']]
    features['pagerank'] = [pagerank_dict[node] for node in features['node']]

    return features


# создаем граф
graph = nx.from_pandas_edgelist(edges, "node_1", "node_2")

# вычисляем признаки для всех узлов
node_features = compute_graph_features(graph=graph)

# объединяем признаки с обучающей выборкой
train_data = pd.merge(train, node_features, on='node', how='left')
test_data = pd.merge(test, node_features, on='node', how='left')

# определяем признаки и целевую переменную
X_train = train_data.drop(['target', 'node'], axis=1)
y_train = train_data['target']
X_test = test_data.drop(['target', 'node'], axis=1)
y_test = test_data['target']

# инициализируем и обучаем модель
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# делаем предсказание на тестовой выборке
y_pred = model.predict(X_test)

# вычисляем MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error на тестовой выборке: {mse:.6f}")  # 0.013555
