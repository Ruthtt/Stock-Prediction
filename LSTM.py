import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import matplotlib.pyplot as plt

# 自定义回调函数，用于计算每轮的ACC和AUC
class MetricsCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        predicted_probabilities = self.model.predict(X_val)
        predicted_classes = (predicted_probabilities > 0.5).astype(int)
        acc = accuracy_score(y_val, predicted_classes)
        auc = roc_auc_score(y_val, predicted_probabilities)
        print(f'Epoch {epoch + 1} - ACC: {acc:.4f}, AUC: {auc:.4f}')

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

data['MA60'] = data['close'].rolling(window=60).mean()  # 60日均线
data['RSI'] = calculate_RSI(data['close'], window=14)  # RSI
data['MACD'] = calculate_MACD(data['close'])  # MACD
data['VROC'] = data['vol'].pct_change(10)  # VROC

# 移除缺失值
data.dropna(inplace=True)

# 3. 数据准备
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[['MA60', 'RSI', 'MACD', 'VROC']])

X = []
y = []
time_step = 60
for i in range(len(scaled_data) - time_step - 1):
    X.append(scaled_data[i:(i + time_step), :])
    # 创建二分类标签：下一个时间步价格高于当前价格为1，否则为0
    if data['close'].iloc[i + time_step] > data['close'].iloc[i + time_step - 1]:
        y.append(1)
    else:
        y.append(0)

X, y = np.array(X), np.array(y)

# 4. 设置交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_no = 1
for train_index, val_index in kf.split(X):
    print(f'Training fold {fold_no} ...')
    
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # 5. 构建LSTM模型
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(time_step, X.shape[2])))
    model.add(Dropout(0.3))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(50))
    model.add(Dense(1, activation='sigmoid'))  # 使用sigmoid激活函数进行二分类

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy')  # 使用binary_crossentropy作为损失函数

    # 6. 模型训练与保存
    checkpoint = ModelCheckpoint(f"best_lstm_model_fold_{fold_no}.keras", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    metrics_callback = MetricsCallback(validation_data=(X_val, y_val))

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping, metrics_callback])

    fold_no += 1

# 7. 评估模型 (使用最后一个fold作为示例)
best_model = load_model(f"best_lstm_model_fold_{fold_no - 1}.keras")
predicted_probabilities = best_model.predict(X_val)
predicted_classes = (predicted_probabilities > 0.5).astype(int)

accuracy = accuracy_score(y_val, predicted_classes)
auc = roc_auc_score(y_val, predicted_probabilities)

print(f'Final Model (Fold {fold_no - 1}) - ACC: {accuracy:.4f}, AUC: {auc:.4f}')

# 绘制预测结果
plt.figure(figsize=(10, 6))
plt.plot(data['trade_date'].iloc[-len(y_val):], y_val, color='blue', label='Actual Classes')
plt.plot(data['trade_date'].iloc[-len(y_val):], predicted_classes, color='red', label='Predicted Classes')
plt.xlabel('Date')
plt.ylabel('Class')
plt.title('Actual vs Predicted Classes')
plt.legend()
plt.show()

# 保存模型
best_model.save(r'D:\python\new\Project_003\final_lstm_model.keras')
