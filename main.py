
import ccxt
import pandas as pd
import numpy as np
import talib
import requests
import openai
import google.cloud.logging
import google.cloud.monitoring_v3
import logging
import time 
from config import API_KEY, API_SECRET, NEWS_API_KEY, OPENAI_API_KEY, assistant_id 


class TradingBot:
    def __init__(self, api_key, api_secret, news_api_key, openai_api_key, assistant_id, project_id):
        self.api_key = api_key
        self.api_secret = api_secret
        self.news_api_key = news_api_key
        self.openai_api_key = openai_api_key
        self.assistant_id = assistant_id
        self.project_id = project_id
        self.exchange = self.initialize_exchange()
        self.data = {}
        self.current_position = None
        self.news_data = []
        self.current_symbol = 'NEAR/USDT'
        self.bitcoin_symbol = 'BTC/USDT'

        # Настройка клиента для логирования
        logging_client = google.cloud.logging.Client()
        logging_client.setup_logging()
        self.logger = logging.getLogger(__name__)

        # Настройка клиента для мониторинга
        self.monitoring_client = google.cloud.monitoring_v3.MetricServiceClient()
        self.project_name = f"projects/{self.project_id}"

    def initialize_exchange(self):
        exchange = ccxt.bybit({
            'apiKey': self.api_key,
            'secret': self.api_secret,
        })
        return exchange

    def fetch_market_data(self, symbol):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='5m')
            self.data[symbol] = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            self.data[symbol]['timestamp'] = pd.to_datetime(self.data[symbol]['timestamp'], unit='ms')
            self.logger.info(f"Fetched market data for {symbol}")
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {e}")

    def calculate_indicators(self, symbol):
        try:
            df = self.data[symbol]
            df['ema50'] = df['close'].ewm(span=50).mean()
            df['ema200'] = df['close'].ewm(span=200).mean()
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            self.data[symbol] = df
            self.logger.info(f"Calculated indicators for {symbol}")
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {e}")

    def analyze_market(self, symbol):
        try:
            df = self.data[symbol]
            last_row = df.iloc[-1]
            buy_signal = (last_row['macd'] > last_row['macd_signal']) and (last_row['rsi'] < 70) and (last_row['close'] > last_row['ema50'])
            sell_signal = (last_row['macd'] < last_row['macd_signal']) and (last_row['rsi'] > 30) and (last_row['close'] < last_row['ema50'])
            self.logger.info(f"Analyzed market for {symbol}")
            return buy_signal, sell_signal
        except Exception as e:
            self.logger.error(f"Error analyzing market for {symbol}: {e}")
            return False, False

    def fetch_news(self):
        try:
            response = requests.get(f'https://newsapi.org/v2/everything?q=NEAR+Protocol&apiKey={self.news_api_key}')
            if response.status_code == 200:
                self.news_data = response.json().get('articles', [])
                self.logger.info("Fetched news data")
            else:
                self.logger.warning(f"Failed to fetch news data: {response.status_code}")
        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")

    def analyze_news(self):
        try:
            positive_keywords = ['partnership', 'investment', 'growth', 'increase', 'positive']
            negative_keywords = ['hack', 'loss', 'decrease', 'negative', 'problem']
            sentiment_score = 0
            for article in self.news_data:
                title = article['title'].lower()
                description = article['description'].lower() if article['description'] else ''
                content = f"{title} {description}"
                for word in positive_keywords:
                    if word in content:
                        sentiment_score += 1
                for word in negative_keywords:
                    if word in content:
                        sentiment_score -= 1
            self.logger.info("Analyzed news data")
            return sentiment_score
        except Exception as e:
            self.logger.error(f"Error analyzing news: {e}")
            return 0

    def analyze_with_openai(self, prompt):
        try:
            openai.api_key = self.openai_api_key
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a trading bot analyzing cryptocurrency markets."},
                    {"role": "user", "content": prompt}
                ],
                assistant_id=self.assistant_id
            )
            self.logger.info("Analyzed data with OpenAI")
            return response.choices[0].message['content'].strip()
        except Exception as e:
            self.logger.error(f"Error analyzing with OpenAI: {e}")
            return ""

    def generate_advice(self):
        try:
            market_data = self.data[self.current_symbol].tail(5).to_dict('records')
            news_summary = " ".join([article['title'] for article in self.news_data])
            prompt = f"Analyze the following market data and news summary for trading advice:\n\nMarket Data:\n{market_data}\n\nNews Summary:\n{news_summary}\n\nProvide detailed trading advice considering the market conditions and news."
            advice = self.analyze_with_openai(prompt)
            self.logger.info("Generated trading advice")
            return advice
        except Exception as e:
            self.logger.error(f"Error generating advice: {e}")
            return ""

    def execute_trade(self, symbol, buy_signal, sell_signal):
        try:
            if buy_signal and self.current_position is None:
                order = self.exchange.create_market_buy_order(symbol, 1)
                self.current_position = 'long'
                self.logger.info(f"Buy order executed for {symbol}: {order}")
            elif sell_signal and self.current_position == 'long':
                order = self.exchange.create_market_sell_order(symbol, 1)
                self.current_position = None
                self.logger.info(f"Sell order executed for {symbol}: {order}")
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")

    def run(self):
        while True:
            self.fetch_market_data(self.current_symbol)
            self.calculate_indicators(self.current_symbol)
            buy_signal, sell_signal = self.analyze_market(self.current_symbol)
            self.fetch_news()
            advice = self.generate_advice()
            self.logger.info(f"Trading advice: {advice}")
            self.execute_trade(self.current_symbol, buy_signal, sell_signal)
            time.sleep(300)  # Ожидание 5 минут до следующего цикла

    def change_symbol(self, new_symbol):
        self.current_symbol = new_symbol
        self.logger.info(f"Trading pair changed to {new_symbol}")

    def stop_trading(self):
        self.current_position = None
        self.logger.info("Trading stopped. Waiting for new entry signals.")

    def search_best_pair(self):
        pairs = ['NEAR/USDT', 'ETH/USDT', 'LTC/USDT']
        best_pair = max(pairs, key=lambda pair: self.exchange.fetch_ticker(pair)['quoteVolume'])
        self.current_symbol = best_pair
        self.logger.info(f"Best trading pair for today: {best_pair}")

    def respond_to_user(self, query):
        if query == "status":
            advice = self.generate_advice()
            self.logger.info(f"Status requested: {advice}")
        elif query.startswith("change_symbol"):
            _, new_symbol = query.split()
            self.change_symbol(new_symbol)
        elif query == "stop_trading":
            self.stop_trading()
        elif query == "search_best_pair":
            self.search_best_pair()
        else:
            self.logger.warning(f"Unknown command: {query}")

# Пример использования
api_key = 'your_api_key'
api_secret = 'your_api_secret'
news_api_key = 'your_news_api_key'
openai_api_key = 'your_openai_api_key'
assistant_id = 'asst_XTwthRzz5YYe9wQyPJ9WRxep'  # Замените на ID вашего ассистента
project_id = 'your_project_id'  # Замените на ID вашего проекта
bot = TradingBot(api_key, api_secret, news_api_key, openai_api_key, assistant_id, project_id)

# Пример выполнения команд
bot.run()  # Запуск основного цикла
# В другом потоке или по событию можно выполнять команды:
# bot.respond_to_user("change_symbol ETH/USDT")
# bot.respond_to_user("stop_trading")
# bot.respond_to_user("search_best_pair")
# bot.respond_to_user("status")

