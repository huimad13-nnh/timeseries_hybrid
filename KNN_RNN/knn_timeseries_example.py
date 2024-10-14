import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

# Tạo dataframe từ tệp giá vàng
df_gld = pd.read_csv("data/goldprice/gld_price_data.csv")
df_gld['Date'] = pd.to_datetime(df_gld['Date'])  # Đảm bảo cột 'Date' là kiểu thời gian

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

def find_optimal_k(X_train, y_train, max_k=30):
    k_range = range(1, max_k+1)
    k_rmse_scores = []
    
    for k in k_range:
        knn = KNeighborsRegressor(n_neighbors=k, metric='manhattan')
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        rmse_scores = -cross_val_score(knn, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
        k_rmse_scores.append(rmse_scores.mean())
    
    # Vẽ biểu đồ RMSE theo giá trị k
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, k_rmse_scores, marker='o')
    plt.xlabel('Giá trị k')
    plt.ylabel('RMSE')
    plt.title('RMSE theo các giá trị k')
    plt.show()
    
    # Tìm giá trị k có RMSE nhỏ nhất
    optimal_k = k_range[np.argmin(k_rmse_scores)]
    print(f'Giá trị k tối ưu: {optimal_k}')
    return optimal_k

# Tìm giá trị k tối ưu
optimal_k = find_optimal_k(X_train, y_train)

# Áp dụng KNN vào dữ liệu chuỗi thời gian
knn = KNeighborsRegressor(n_neighbors=optimal_k)
knn.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = knn.predict(X_test)

# Tính Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

# Tính MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae}")

#Tính MAPE
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
mape = calculate_mape(y_test, y_pred)
print(f"MAPE: {mape}")


# Vẽ biểu đồ kết quả
plt.figure(figsize=(10, 6))
plt.plot(time[window_size:], data[window_size:], label='Dữ liệu thực', color='blue')
plt.plot(time[split_idx+window_size:], y_pred, label='Dự đoán KNN', color='red', linestyle='--')
plt.title('Dự đoán chuỗi thời gian với KNN và cửa sổ trượt')
plt.xlabel('Thời gian')
plt.ylabel('Giá Vàng')
plt.legend()
plt.show()