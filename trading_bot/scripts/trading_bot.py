import gym
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from keras.models import load_model
import time
import logging
import telegram
import requests
from sklearn.preprocessing import MinMaxScaler
import schedule

# Настройка логирования и Telegram
logging.basicConfig(filename='trading_bot.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
bot = telegram.Bot(token='YOUR_TELEGRAM_BOT_TOKEN')
chat_id = 'YOUR_CHAT_ID'

def send_telegram_message(message):
    bot.send_message(chat_id=chat_id, text=message)

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

def get_binance_historical_data(symbol="BTCUSDT", interval="1h", limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def preprocess_data(data, look_back=1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - look_back - 1):
        a = scaled_data[i:(i + look_back), 0]
        X.append(a)
        y.append(scaled_data[i + look_back, 0])
    
    return np.array(X), np.array(y), scaler

def load_lstm_model():
    return load_model('../models/lstm_model.h5')

def predict_trade(model, data, scaler, look_back=10):
    data = scaler.transform(data)
    X = np.array([data[-look_back:]])
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    prediction = model.predict(X)
    return scaler.inverse_transform(prediction)[0][0]

def execute_trade(action, current_price):
    if action == 0:  # Buy
        logging.info("Executing BUY order")
        send_telegram_message("Executing BUY order")
        stop_loss = current_price * 0.98
        take_profit = current_price * 1.02
        logging.info(f"Stop Loss set at {stop_loss}, Take Profit set at {take_profit}")
    elif action == 2:  # Sell
        logging.info("Executing SELL order")
        send_telegram_message("Executing SELL order")
        stop_loss = current_price * 1.02
        take_profit = current_price * 0.98
        logging.info(f"Stop Loss set at {stop_loss}, Take Profit set at {take_profit}")

# Загрузка моделей
model_lstm = load_lstm_model()
model_rl = PPO.load('../models/ppo_crypto_trading')

# Основной цикл торговли
def trading_loop():
    latest_data = get_binance_historical_data()
    current_price = latest_data['close'].values[-1]
    prediction_price = predict_trade(model_lstm, latest_data[['close']].values, scaler, look_back)
    action_rl, _ = model_rl.predict(latest_data.values)
    execute_trade(action_rl, current_price)
    logging.info(f"Current Price: {current_price}, Predicted Price: {prediction_price}, Action: {'BUY' if action_rl == 0 else 'SELL' if action_rl == 2 else 'HOLD'}")
    send_telegram_message(f"Current Price: {current_price}, Predicted Price: {prediction_price}, Action: {'BUY' if action_rl == 0 else 'SELL' if action_rl == 2 else 'HOLD'}")

# Запуск торговли каждый час
schedule.every().hour.do(trading_loop)

# Основной цикл для выполнения запланированных задач
while True:
    schedule.run_pending()
    time.sleep(1)
