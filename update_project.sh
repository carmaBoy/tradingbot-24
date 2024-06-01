#!/bin/bash

# Перейти в директорию проекта
cd ~/Python-3.9.13/tradingbot-24

# Получить последние изменения из репозитория
git pull origin main

# Установка новых или обновленных зависимостей
source venv/bin/activate
pip install -r requirements.txt

# Запуск проекта
python main.py
