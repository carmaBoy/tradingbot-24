import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Загрузка данных
X = np.load('../data/X.npy')
y = np.load('../data/y.npy')

# Формирование данных для LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Создание модели
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Обучение модели
model.fit(X, y, epochs=10, batch_size=1, verbose=2)

# Сохранение модели
model.save('../models/lstm_model.h5')
