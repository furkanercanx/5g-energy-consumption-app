import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from model_5g import predict_energy

# Streamlit sayfa ayarları
st.set_page_config(
    page_title="5G Energy Consumption App",
    page_icon="📡",
    layout="wide"
)

# ------------------------------
# Yardımcı fonksiyonlar
# ------------------------------
@st.cache_resource
def load_5g_model():
    """
    Eğitilmiş modeli ve scaler'ı models klasöründen yükler.
    """
    model_path = os.path.join("models", "5g_energy_model.pkl")
    scaler_path = os.path.join("models", "5g_feature_scaler.pkl")
    features_path = os.path.join("models", "5g_feature_names.pkl")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path)):
        st.error("Model dosyaları bulunamadı. Lütfen 'models' klasörünü kontrol edin.")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(features_path, "rb") as f:
        feature_names = pickle.load(f)

    return model, scaler, feature_names


@st.cache_data
def load_5g_data():
    """
    Veri setini okur, hour ve BS_id kolonlarını ekler.
    """
    df = pd.read_csv("5G_energy_consumption_dataset.csv")

    # Saat bilgisi (Time: '20230101 010000' gibi)
    df["hour"] = df["Time"].str.slice(9, 11).astype(int)

    # Baz istasyonu numarası (B_0 -> 0)
    df["BS_id"] = df["BS"].str.extract(r"B_(\d+)").astype(int)

    return df


# ------------------------------
# Ana Streamlit uygulaması
# ------------------------------
def main():
    st.title("🎁 5G Energy Consumption Gift App")
    st.markdown("""
Bu uygulama, seçtiğin **trafik yükü (load)**, **enerji modu (ESMODE)**, 
**TXpower**, **saat** ve **baz istasyonu (BS)** bilgilerine göre 
tahmini enerji tüketimini hesaplar.
    """)

    with st.spinner("Model ve veri yükleniyor..."):
        model, scaler, feature_names = load_5g_model()
        df = load_5g_data()

    # ---------------- Sidebar (kullanıcı girişleri) ----------------
    st.sidebar.header("Girdi Parametreleri")

    # Baz istasyonu seçimi
    bs_list = sorted(df["BS"].unique())
    selected_bs = st.sidebar.selectbox("Baz İstasyonu (BS)", bs_list)
    bs_id = int(selected_bs.split("_")[1])  # B_0 -> 0

    # load (0-1 arası)
    load_val = st.sidebar.slider("Trafik Yükü (load)", 0.0, 1.0, 0.3, 0.01)

    # ESMODE (0-4 arası integer)
    esmode_val = st.sidebar.selectbox("Enerji Tasarruf Modu (ESMODE)", [0, 1, 2, 3, 4])

    # TXpower için min-max
    tx_min = float(df["TXpower"].min())
    tx_max = float(df["TXpower"].max())
    tx_mean = float(df["TXpower"].mean())
    txpower_val = st.sidebar.slider("TXpower", tx_min, tx_max, tx_mean, 0.01)

    # Saat seçimi
    hour_val = st.sidebar.slider("Saat (0-23)", 0, 23, 12, 1)

    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("⚡ Enerji Tüketimini Tahmin Et")

    # ---------------- Tahmin bölümü ----------------
    if predict_button:
        # Modelin beklediği sıraya göre feature dictionary
        features_dict = {
            "load": load_val,
            "ESMODE": esmode_val,
            "TXpower": txpower_val,
            "hour": hour_val,
            "BS_id": bs_id
        }

        # Tahmin
        pred_energy = predict_energy(model, scaler, feature_names, features_dict)

        # Bunu % skala gibi göstermek için 0–100 aralığına kırpıyoruz (sadece görselleştirme)
        energy_percent = max(0, min(100, pred_energy))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Tahmini Enerji Tüketimi")
            st.metric("Anlık Enerji Kullanımı", f"%{energy_percent:.1f}")

            if energy_percent < 30:
                level = "🔵 Düşük"
            elif energy_percent < 60:
                level = "🟡 Orta"
            else:
                level = "🔴 Yüksek"

            st.write(f"Seviye: **{level}**")
            st.progress(int(energy_percent))

        with col2:
            st.subheader(f"{selected_bs} Baz İstasyonu Özeti")
            bs_data = df[df["BS"] == selected_bs]
            bs_mean = bs_data["Energy"].mean()
            st.write(f"- Geçmiş ortalama enerji: **{bs_mean:.2f}**")
            st.write(f"- Gözlem sayısı: **{len(bs_data)}**")

            if pred_energy > bs_mean:
                st.info("Bu konfigürasyon, bu baz istasyonu ortalamasının **ÜZERİNDE** bir tüketim üretiyor.")
            else:
                st.info("Bu konfigürasyon, bu baz istasyonu ortalamasının **ALTINDA** daha verimli görünüyor.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info("""
Bu uygulama, 5G baz istasyonlarının enerji tüketimini tahmin etmek için 
Linear Regression tabanlı bir makine öğrenmesi modeli kullanır.
    """)


if __name__ == "__main__":
    main()
