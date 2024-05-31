import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data, look_back=1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(scaled_data) - look_back - 1):
        a = scaled_data[i:(i + look_back), 0]
        X.append(a)
        y.append(scaled_data[i + look_back, 0])
    
    return np.array(X), np.array(y), scaler

if __name__ == "__main__":
    historical_data = pd.read_csv('../data/historical_data.csv')
    data = historical_data[['close']].values
    look_back = 10
    X, y, scaler = preprocess_data(data, look_back)
    np.save('../data/X.npy', X)
    np.save('../data/y.npy', y)
    np.save('../data/scaler.npy', scaler)
