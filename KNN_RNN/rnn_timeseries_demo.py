import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import SimpleRNN, Dense # type: ignore

# Đọc dữ liệu
df_gld = pd.read_csv("data/goldprice/gld_price_data.csv")

# Đảm bảo cột 'Date' là kiểu thời gian và loại bỏ giá trị bị thiếu
df_gld['Date'] = pd.to_datetime(df_gld['Date'])

# Sử dụng giá vàng 'GLD' làm chuỗi thời gian cho dự đoán
data = df_gld[['GLD']].values

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

split_ratio = 0.8
split_idx = int(len(scaled_data) * split_ratio)
data_train, data_test = scaled_data[:split_idx], scaled_data[split_idx:]

# Chuẩn bị dữ liệu theo chuỗi thời gian: tạo hàm create dataset cho ra 2 mảng 1 là chuỗi con từ 1 tới time_step và 1 mảng là chuỗi 
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)



# Chia dữ liệu thành chuỗi thời gian với mỗi bước là 5 ngày
time_step = 5
X_train, y_train = create_dataset(data_train, time_step)
X_test, y_test = create_dataset(data_test, time_step)
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape dữ liệu để phù hợp với mô hình SimpleRNN [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
y_train = np.reshape(y_train, (y_train.shape[0],1))
print("X_train :",X_train.shape,"y_train :",y_train.shape)

#Xây dựng mô hình SimpleRNN
model = Sequential()
model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(SimpleRNN(units=50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile mô hình
model.compile(optimizer='adam', loss='mean_squared_error')

# Huấn luyện mô hình
model.fit(X_train, y_train, batch_size = 32, epochs = 5)

# Dự đoán và đánh giá
# Dự đoán trên tập dữ liệu
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
predicted_prices = np.reshape(predicted_prices, -1)


acctually_prices = data[split_idx+time_step+1:]
acctually_prices = np.reshape(acctually_prices, -1)

# Tính Mean Squared Error
mse = mean_squared_error(acctually_prices, predicted_prices)
print(f"MSE: {mse}")

# Tính MAE
mae = mean_absolute_error(acctually_prices, predicted_prices)
print(f"MAE: {mae}")

#Tính MAPE
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
mape = calculate_mape(acctually_prices, predicted_prices)
print(f"MAPE: {mape}")

# Vẽ biểu đồ so sánh giá trị thực tế và giá trị dự đoán
plt.figure(figsize=(10, 6))
# # plt.plot(df_gld['Date'][-len(predicted_prices):], data[-len(predicted_prices):], label='Giá trị thực tế (GLD)', color='blue')
# # plt.plot(df_gld['Date'][-len(predicted_prices):], predicted_prices, label='Giá trị dự đoán (SimpleRNN)', color='red', linestyle='--')

# # plt.title('So sánh giá trị thực tế và dự đoán của giá vàng (GLD)')
# # plt.xlabel('Ngày')
# # plt.ylabel('Giá vàng (GLD)')
# # plt.legend()
# # plt.grid(True)
# # plt.show()
plt.plot(df_gld["Date"][time_step:] , data[time_step:],label = "train_data", color = "b")
plt.plot(df_gld["Date"][time_step+split_idx+1:], predicted_prices, color = "red", linestyle = '--')
plt.legend()
plt.show()