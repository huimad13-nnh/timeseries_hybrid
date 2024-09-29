import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


df_gld = pd.read_csv("data/goldprice/gld_price_data.csv") # đọc file csv

df_gld['Date'] = pd.to_datetime(df_gld['Date']) # Đảm bảo cột 'Date' là kiểu thời gian
df_gld_clean = df_gld.dropna() # lọc dữ liệu bị thiếu

X = df_gld_clean[['SPX']].values
y = df_gld_clean['GLD'].values
dates = df_gld_clean['Date'].values  # Lấy cột ngày để vẽ biểu đồ


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% huấn luyện, 20% kiểm tra)
dates_train, dates_test = train_test_split(df_gld_clean['Date'].values, test_size=0.2, shuffle=False)  


#Chuẩn hóa dữ liệu
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsRegressor(n_neighbors=9)

knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

# Tính RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Hiển thị kết quả
print(f"Root Mean Squared Error (RMSE): {rmse}")


plt.figure(figsize=(10, 6))
plt.plot(dates_test, y_test, label='Giá trị thực tế (GLD)', color='blue')
plt.plot(dates_test, y_pred, label='Giá trị dự đoán (KNN)', color='red', linestyle='--')

plt.title('So sánh giá trị thực tế và dự đoán của giá vàng (GLD)')
plt.xlabel('Ngày')
plt.ylabel('Giá vàng (GLD)')
plt.legend()
plt.grid(True)
plt.show()