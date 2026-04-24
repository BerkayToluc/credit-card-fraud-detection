import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import joblib
import os

def main():
    # Dosya yolu
    data_path = 'creditcard.csv'
    
    if not os.path.exists(data_path):
        print(f"Hata: '{data_path}' bulunamadı.")
        return

    print("Veri seti yükleniyor...")
    df = pd.read_csv(data_path)
    
    # Amount ve Time sütunlarını standartlaştırma
    print("Veri ön işleme: 'Amount' ve 'Time' sütunları standartlaştırılıyor...")
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    # Hedef değişken (Class) ve özellikleri (Features) ayırma
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Eğitim ve test setlerine ayırma (%80 eğitim, %20 test)
    # stratify=y kullanılarak sınıfların (frauds vs normal) dengeli dağılması sağlanıyor
    print("Veri seti eğitim ve test olarak bölünüyor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Random Forest modelini tanımlama ve eğitme (Hız için n_estimators=50 ve tüm çekirdekler n_jobs=-1)
    print("Model eğitiliyor (Random Forest)... Bu işlem biraz zaman alabilir.")
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Test verisi ile tahmin yapma
    print("Test seti üzerinde tahmin yapılıyor...")
    y_pred = model.predict(X_test)
    
    # Performans metriklerini hesaplama
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("             MODEL PERFORMANS METRİKLERİ")
    print("="*50)
    print(f"Precision : {precision:.4f} (Dolandırıcı dediğimizin ne kadarı gerçekten dolandırıcı)")
    print(f"Recall    : {recall:.4f} (Gerçek dolandırıcıların ne kadarını bulabildik)")
    print(f"F1-Score  : {f1:.4f} (Precision ve Recall değerlerinin harmonik ortalaması)")
    print("="*50)
    
    print("\nDetaylı Sınıflandırma Raporu (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Dolandırıcı (1)"]))
    
    # Modeli ileride kullanmak üzere .pkl dosyası olarak kaydetme
    model_filename = 'fraud_detection_model.pkl'
    joblib.dump(model, model_filename)
    print(f"\nBaşarılı: Eğitilmiş model '{model_filename}' olarak kaydedildi!")

if __name__ == "__main__":
    main()
