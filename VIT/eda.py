import numpy as np
from sklearn.preprocessing import MinMaxScaler

X = np.load('../data/X_train.npy')
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = X / 255

print(X_scaled.max(), X_scaled.min())
