import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Eksik verileri temizle
    data = data.dropna()

    # Zaman bilgisinden saat çıkar
    data["hour"] = data["Time"].str.slice(9, 11).astype(int)

    # Baz istasyonu numarasını çıkar (B_0 → 0)
    data["BS_id"] = data["BS"].str.extract(r"B_(\d+)").astype(int)

    # Modelin kullanacağı özellikler
    feature_cols = ["load", "ESMODE", "TXpower", "hour", "BS_id"]

    X = data[feature_cols]
    y = data["Energy"]

    # Veri setini eğitim ve test olarak ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalizasyon
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test R²: {r2:.4f}")

    return rmse, r2

def predict_energy(model, scaler, feature_names, features_dict):
    # Kullanıcının verdiği özelliklerden bir satır oluştur
    row = [features_dict[f] for f in feature_names]
    row = np.array(row).reshape(1, -1)

    # Normalizasyon uygula
    row_scaled = scaler.transform(row)

    # Tahmin
    pred = model.predict(row_scaled)[0]

    return pred

if __name__ == "__main__":
    # Test amaçlı: model eğitilip performans yazdırılacak
    data = load_data("5G_energy_consumption_dataset.csv")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols = preprocess_data(data)
    model = train_model(X_train_scaled, y_train)
    evaluate_model(model, X_test_scaled, y_test)
