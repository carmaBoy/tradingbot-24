import gym
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class CryptoTradingEnv(gym.Env):
    def __init__(self, df):
        self.df = df
        self.action_space = gym.spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(df.columns),), dtype=np.float16)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.df.iloc[self.current_step].values
        return obs

    def step(self, action):
        self.current_step += 1
        reward = 0
        done = False
        
        if action == 0:  # Buy
            reward = self.df['close'].iloc[self.current_step] - self.df['close'].iloc[self.current_step - 1]
        elif action == 2:  # Sell
            reward = self.df['close'].iloc[self.current_step - 1] - self.df['close'].iloc[self.current_step]
        
        if self.current_step >= len(self.df) - 1:
            done = True
        
        obs = self._next_observation()
        return obs, reward, done, {}

# Загрузка данных
historical_data = pd.read_csv('../data/historical_data.csv')

# Создание окружения
env = DummyVecEnv([lambda: CryptoTradingEnv(historical_data)])

# Обучение модели с подкреплением
model_rl = PPO('MlpPolicy', env, verbose=1)
model_rl.learn(total_timesteps=10000)
model_rl.save('../models/ppo_crypto_trading')  # Сохранение модели
