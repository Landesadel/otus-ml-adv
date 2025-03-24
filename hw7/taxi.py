import numpy as np
import gymnasium as gym
import time
from pathlib import Path

# Гипперпараметры
EPISODES = 100000
EVAL_INTERVAL = 500
ALPHA = 0.7  # Начальная скорость обучения
ALPHA_DECAY = 0.99995  # Затухание скорости обучения
GAMMA = 0.95  # Коэффициент дисконтирования
EPSILON = 1.0  # Начальная вероятность исследования
EPSILON_DECAY = 0.99995  # Скорость затухания исследования
MIN_EPSILON = 0.01  # Минимальный epsilon
MIN_ALPHA = 0.1  # Минимальная скорость обучения

# Инициализация среды
env = gym.make('Taxi-v3')
Q = np.zeros((env.observation_space.n, env.action_space.n))

rewards_history = []
episode_lengths = []
best_reward = float('-inf')

# Обучение с отслеживанием прогресса
for episode in range(EPISODES):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    steps = 0

    while not done:
        # ε-жадная стратегия
        if np.random.uniform() < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        # Шаг среды
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Обновление Q-таблицы с векторными операциями
        old_value = Q[state, action]
        next_max = np.max(Q[next_state])

        # Новый Q-value
        new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
        Q[state, action] = new_value

        state = next_state
        episode_reward += reward
        steps += 1

    # Затухание параметров
    EPSILON = max(EPSILON * EPSILON_DECAY, MIN_EPSILON)
    ALPHA = max(ALPHA * ALPHA_DECAY, MIN_ALPHA)

    # Сохранение истории
    rewards_history.append(episode_reward)
    episode_lengths.append(steps)

    # Оценка и сохранение лучшей модели
    if episode_reward > best_reward:
        best_reward = episode_reward
        np.save("best_q_table.npy", Q)

    # Вывод статистики
    if (episode + 1) % EVAL_INTERVAL == 0:
        avg_reward = np.mean(rewards_history[-EVAL_INTERVAL:])
        avg_length = np.mean(episode_lengths[-EVAL_INTERVAL:])
        print(f"Эпизод {episode + 1}/{EPISODES}")
        print(f"Средняя награда: {avg_reward:.2f}")
        print(f"Средняя длина эпизода: {avg_length:.2f}")
        print(f"Текущий epsilon: {EPSILON:.4f}")
        print(f"Текущий alpha: {ALPHA:.4f}")
        print("-----------------------")

# Загрузка лучшей модели
if Path("best_q_table.npy").exists():
    Q = np.load("best_q_table.npy")

# Финальное тестирование
test_episodes = 1000
success_count = 0
total_rewards = []

for _ in range(test_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward

    total_rewards.append(episode_reward)
    if episode_reward > 0:  # Предполагаем, что положительная награда = успех
        success_count += 1

print("\nФинальные результаты:")
print(f"Успешных эпизодов: {success_count}/{test_episodes} ({(success_count / test_episodes) * 100:.2f}%)")
print(f"Средняя награда: {np.mean(total_rewards):.2f}")

# Визуализация с улучшениями
env = gym.make("Taxi-v3", render_mode="human")
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = np.argmax(Q[state])
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward

    # Очистка вывода для анимации
    print("\033c", end='')
    env.render()
    print(f"Текущее действие: {action}")
    print(f"Накопленная награда: {total_reward}")
    time.sleep(0.5)

    state = next_state

env.close()
