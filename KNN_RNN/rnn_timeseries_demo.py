import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import SimpleRNN, Dense

# Đọc dữ liệu
df_gld = pd.read_csv("data/goldprice/gld_price_data.csv")

# Đảm bảo cột 'Date' là kiểu thời gian và loại bỏ giá trị bị thiếu
df_gld['Date'] = pd.to_datetime(df_gld['Date'])
df_gld_clean = df_gld.dropna()

# Sử dụng giá vàng 'GLD' làm chuỗi thời gian cho dự đoán
data = df_gld_clean[['GLD']].values

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Chuẩn bị dữ liệu theo chuỗi thời gian
def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Chia dữ liệu thành chuỗi thời gian với mỗi bước là 60 ngày
time_step = 60
X, y = create_dataset(scaled_data, time_step)


# Reshape dữ liệu để phù hợp với mô hình SimpleRNN [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)
print(X)

#Xây dựng mô hình SimpleRNN
model = Sequential()
model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(SimpleRNN(units=50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile mô hình
model.compile(optimizer='adam', loss='mean_squared_error')

# Huấn luyện mô hình
model.fit(X, y, batch_size=64, epochs=10)

# Dự đoán và đánh giá
# Dự đoán trên tập dữ liệu
predicted_prices = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Tính RMSE
rmse = np.sqrt(mean_squared_error(data[time_step+1:], predicted_prices))
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Vẽ biểu đồ so sánh giá trị thực tế và giá trị dự đoán
plt.figure(figsize=(10, 6))
plt.plot(df_gld_clean['Date'][-len(predicted_prices):], data[-len(predicted_prices):], label='Giá trị thực tế (GLD)', color='blue')
plt.plot(df_gld_clean['Date'][-len(predicted_prices):], predicted_prices, label='Giá trị dự đoán (SimpleRNN)', color='red', linestyle='--')

plt.title('So sánh giá trị thực tế và dự đoán của giá vàng (GLD)')
plt.xlabel('Ngày')
plt.ylabel('Giá vàng (GLD)')
plt.legend()
plt.grid(True)
plt.show()