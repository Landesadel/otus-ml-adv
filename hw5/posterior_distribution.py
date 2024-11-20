import numpy as np
from scipy.stats import beta, bernoulli
import seaborn as sns
import matplotlib.pyplot as plt


N = 100         # Количество подбрасываний в наблюдении
H = 58          # Количество орлов в наблюдении
alpha = 1       # Параметр альфа для априорного Beta распределения
beta_param = 1  # Параметр бета для априорного Beta распределения
k = 40          # Размер выборки апостериорного распределения
M = 10          # Количество предсказательных подбрасываний для каждого p

# Вычисляем параметры апостериорного распределения
alpha_post = alpha + H
beta_post = beta_param + N - H

# Генерируем выборки из апостериорного распределения
posterior_samples = beta.rvs(alpha_post, beta_post, size=k)

# для корректного broadcast добавляем новую ось к posterior_samples
p_matrix = posterior_samples[:, np.newaxis]  # размерность (40, 1)

predictive_samples = bernoulli.rvs(p=p_matrix, size=(k, M))  # размерность (40, 10)

# суммируем кол-во орлов в каждом наборе М подбрасываний
predictive_sums = predictive_samples.sum(axis=1)

# строим гистограмму предсказательного распределения
plt.figure(figsize=(10,6))
sns.histplot(predictive_sums, bins=range(0, M+2), kde=False, stat='density', color='skyblue', edgecolor='black')
plt.title(f'Апостериорное предсказательное распределение количества орлов в {M} подбрасываниях')
plt.xlabel('Количество орлов')
plt.ylabel('Плотность вероятности')
plt.xticks(range(0, M+1))
plt.show()
