import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 1. 加载数据
file_path = r'D:\python\new\Project_003\all_stock_data_last_year.csv'
data = pd.read_csv(file_path)

# 2. 计算技术指标
def calculate_RSI(series, window=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_MACD(series, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = series.ewm(span=fast_period, min_periods=slow_period).mean()
    ema_slow = series.ewm(span=slow_period, min_periods=slow_period).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, min_periods=signal_period).mean()
    return macd - signal

data['MA'] = data['close'].rolling(window=20).mean()  # 20日均线
data['RSI'] = calculate_RSI(data['close'], window=14)  # RSI
data['MACD'] = calculate_MACD(data['close'])  # MACD
data['VROC'] = data['vol'].pct_change(10)  # VROC

# 移除缺失值
data.dropna(inplace=True)

# 3. 数据准备
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['MA', 'RSI', 'MACD', 'VROC']])

X = []
y = []
time_step = 60
for i in range(len(scaled_data) - time_step - 1):
    X.append(scaled_data[i:(i + time_step), :])
    y.append(data['close'].iloc[i + time_step])

X, y = np.array(X), np.array(y)
y = y.reshape(-1, 1)
y = scaler.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. 构建LSTM模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. 模型训练与保存
checkpoint = ModelCheckpoint("best_lstm_model.keras", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint, early_stopping])

# 6. 评估模型
best_model = load_model("best_lstm_model.keras")
predicted = best_model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y_test)

# 绘制预测结果
plt.figure(figsize=(10, 6))
plt.plot(data['trade_date'].iloc[-len(y_test):], actual, color='blue', label='Actual Price')
plt.plot(data['trade_date'].iloc[-len(y_test):], predicted, color='red', label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()

# 保存模型
best_model.save(r'D:\python\new\Project_003\final_lstm_model.keras')
