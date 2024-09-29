import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

# Tạo dataframe từ tệp giá vàng
df_gld = pd.read_csv("data/goldprice/gld_price_data.csv")
df_gld['Date'] = pd.to_datetime(df_gld['Date'])  # Đảm bảo cột 'Date' là kiểu thời gian
df_gld['GLD'].fillna(df_gld['GLD'].mode()[0], inplace=True) # Điền giá trị thiếu 'NA' bằng giá trị xuất hiện nhiều nhất

# Lấy dữ liệu cột 'GLD' và 'Date'
data = df_gld['GLD'].values
time = df_gld['Date'].values

# Định nghĩa hàm tạo các cửa sổ trượt
def create_sliding_windows(data, window_size):
    X, y = [], [] #khởi tạo tập rỗng
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# Thiết lập kích thước cửa sổ
window_size = 5

# Tạo các cửa sổ trượt
X, y = create_sliding_windows(data, window_size)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
split_ratio = 0.8
split_idx = int(len(X) * split_ratio)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Áp dụng KNN vào dữ liệu chuỗi thời gian
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = knn.predict(X_test)

# Vẽ biểu đồ kết quả
plt.figure(figsize=(10, 6))
plt.plot(time[window_size:], data[window_size:], label='Dữ liệu thực', color='blue')
plt.plot(time[split_idx+window_size:], y_pred, label='Dự đoán KNN', color='red', linestyle='--')
plt.title('Dự đoán chuỗi thời gian với KNN và cửa sổ trượt')
plt.xlabel('Thời gian')
plt.ylabel('Giá Vàng')
plt.legend()
plt.show()