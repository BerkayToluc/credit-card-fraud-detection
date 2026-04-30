import streamlit as st
import pandas as pd
import joblib
import random
from sklearn.preprocessing import StandardScaler

# Sayfa yapılandırması
st.set_page_config(page_title="Dolandırıcılık Tespit Sistemi", layout="wide")

# Veri yükleme ve ön işleme (Önbelleğe alınmış)
@st.cache_data
def load_data():
    # Veri setini okuma
    df_raw = pd.read_csv('creditcard.csv')
    
    # Modelin eğitildiği gibi Amount ve Time sütunlarını standartlaştıralım
    df_processed = df_raw.copy()
    scaler = StandardScaler()
    df_processed['Amount'] = scaler.fit_transform(df_processed['Amount'].values.reshape(-1, 1))
    df_processed['Time'] = scaler.fit_transform(df_processed['Time'].values.reshape(-1, 1))
    
    return df_raw, df_processed

# Modeli yükleme (Önbelleğe alınmış)
@st.cache_resource
def load_model():
    return joblib.load('fraud_detection_model.pkl')

def process_transaction(target_class, df_raw, df_processed, model):
    # İlgili sınıfa ait verileri filtrele
    subset_indices = df_raw[df_raw['Class'] == target_class].index
    
    if len(subset_indices) == 0:
        st.warning("Bu sınıfa ait veri bulunamadı.")
        return
        
    # Rastgele bir indeks seç
    random_index = random.choice(subset_indices)
    
    # Gerçek tutarı (Amount) ve Time değerini df_raw'dan al
    real_amount = df_raw.loc[random_index, 'Amount']
    real_time = df_raw.loc[random_index, 'Time']
    
    st.subheader("İşlem Detayları")
    
    # Tutar ve Time değerini yan yana şık bir metrik kartı olarak göster
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="İşlem Tutarı (Amount)", value=f"${real_amount:,.2f}")
    with col2:
        st.metric(label="İşlem Zamanı (Time)", value=f"{real_time:.0f}")
        
    st.markdown("**İşleme Ait Tüm Özellikler (Arka Plan Verisi):**")
    
    # Seçilen satırın tüm özelliklerini DataFrame olarak göster
    # Hoca şeffafça görebilsin diye df_raw'dan class sütunu hariç hepsini alıyoruz
    features_to_display = df_raw.loc[[random_index]].drop('Class', axis=1)
    st.dataframe(features_to_display, use_container_width=True)
    
    st.markdown("---")
    
    # Model için özellikleri df_processed'dan al (Class hariç)
    # Özellik isimlerinin korunması için DataFrame olarak gönderiyoruz
    features_df = df_processed.drop('Class', axis=1).loc[[random_index]]
    
    # Modele tahmin yaptır ve olasılıkları hesapla
    prediction = model.predict(features_df)[0]
    probabilities = model.predict_proba(features_df)[0]
    
    # Modelin bu karardan yüzde kaç emin olduğu (Güven Skoru)
    confidence_score = probabilities[prediction] * 100
    
    st.subheader("Model Tahmini")
    if prediction == 0:
        st.success(f"✅ GÜVENLİ İŞLEM ONAYLANDI (Modelin Eminlik Oranı: %{confidence_score:.0f})")
    else:
        st.error(f"🚨 DİKKAT: DOLANDIRICILIK ŞÜPHESİ! (Modelin Eminlik Oranı: %{confidence_score:.0f})")

def main():
    # --- Sol Menü (Sidebar) ---
    st.sidebar.title("Kredi Kartı Dolandırıcılık Tespit Sistemi")
    st.sidebar.markdown("Bu sistem, finansal işlem verilerini analiz ederek makine öğrenimi algoritmalarıyla anormallik tespiti yapmaktadır.")
    st.sidebar.markdown("---")
    
    # Veri ve modeli yükle (Ana ekranda yükleme animasyonu göstermek için)
    try:
        with st.spinner("Veri seti ve model yükleniyor..."):
            df_raw, df_processed = load_data()
            model = load_model()
    except Exception as e:
        st.error(f"Veri veya model yüklenirken bir hata oluştu: {e}")
        st.info("Lütfen 'creditcard.csv' ve 'fraud_detection_model.pkl' dosyalarının uygulama ile aynı klasörde olduğundan emin olun.")
        return

    st.sidebar.markdown("### Bir İşlem Seçin")
    
    # Butonlar sol menüde
    btn_normal = st.sidebar.button("Rastgele Normal İşlem Getir", use_container_width=True)
    btn_fraud = st.sidebar.button("Rastgele Dolandırıcılık İşlemi Getir", use_container_width=True)
    
    # --- Ana Ekran ---
    if btn_normal:
        process_transaction(0, df_raw, df_processed, model)
    elif btn_fraud:
        process_transaction(1, df_raw, df_processed, model)
    else:
        # Başlangıç durumu
        st.info("👈 Lütfen sol menüden analiz etmek istediğiniz bir işlem türü seçerek başlayın.")

if __name__ == "__main__":
    main()
