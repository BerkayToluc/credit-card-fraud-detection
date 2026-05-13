# Kredi Kartı Dolandırıcılık Tespit Merkezi (Fraud Detection)
Bu projede finansal işlem verilerini analiz ederek kredi kartı işleminin yasal/gerçek mi yoksa dolandırıcılık/sahte mi olduğunu tahmin eden Binary Classification tabanlı  bir web uygulaması geliştirdim. 
Sistemin arka planında en az %90 doğruluk payı verem gelişmiş bir yapay zeka modeli ile çalışırken, ön yüzde sekmeli ve modern bir Streamlit arayüzü tercih ettim.

## Proje Geliştirme Adımları ve Özellikler
* **Algoritma Seçimi:** Modelin eğitimi için XGBoost (XGBClassifier) algoritması kullandım. Daha hızlı ve isabetli sonuç verdiği için tercih ettim.
* **Veri Dengeleme (SMOTE):** Kredi kartı veri setindeki dolandırıcılık işlemleri çok az olduğu için veri seti dengesizdi. Sınıf dengesizliğini çözmek ve modelin sahte işlemleri ezberlemesini önlemek amacıyla eğitim verisine SMOTE algoritması uygulandı.
* **Streamlit Arayüzü:** Uygulama üç ana sekmeden oluşmaktadır: Anlık tahminlerin yapıldığı analiz paneli, o anki oturumda yapılan sorguların tutulduğu geçmiş tablosu ve modelin karar aşamalarının gösterildiği grafik bölümü.
* **Özellik Önemi (Feature Importance):** Modelin siyah kutu (black box) olmaktan çıkması için XGBoost'un feature importances özelliği kullanıldı. Uygulama üzerinden, modelin tahmin yaparken en çok hangi değişkenlerden etkilendiğimi grafiksel olarak gösterdim.

  ## Kod Mimarisi ve Kalitesi (OOP & Clean Code)
Projeyi, **Nesne Yönelimli Programlama (OOP)** yaklaşımıyla geliştirdim:
* Kodlar `FraudDetectionTrainer` ve `FraudDetectionApp` gibi sınıflar (Class) altında modüler hale getirdim.
* Tüm metotlarda Google stili **Docstring** açıklamaları ve Python **Type Hint** (tip belirteçleri) kullanılarak okunabilirlik maksimize ettim.

* ## Kullanılan Teknolojiler ve Kütüphaneler
 Python, Pandas, Scikit-Learn, XGBoost, imbalanced-learn (SMOTE), Streamlit, Altair

* ## Proje Dosyaları
* Proje karmaşıklıktan uzak, doğrudan amaca yönelik iki ana Python dosyasından oluşmaktadır:
* `train_model.py`: Veri setinin okunduğu, StandardScaler ile ölçeklendirildiği, SMOTE ile dengelendiği ve XGBoost modelinin eğitilip kaydedildiği arka plan dosyasıdır.
* `app.py`: Kullanıcının etkileşime girdiği Streamlit web arayüzü dosyasıdır.
* `fraud_detection_model.pkl`: `train_model.py` tarafından eğitilip dışa aktarılan ve arayüzün tahmin yaparken kullandığı makine öğrenimi modelidir.
* `creditcard.csv`: (Boyutu nedeniyle GitHub'a yükleyemedim bu dosyayı) Modelin eğitildiği Kaggle veri setidir.

* ## Modelleme Süreci
- Veri setindeki "Time" ve "Amount" değişkenleri, modelin daha sağlıklı çalıştırılabilmesi için `StandardScaler` kullanarak standartlaştırılmıştır.
- Veri seti %80 eğitim ve %20 test olarak ikiye ayrılmıştır.
- Eğitim setindeki dolandırıcılık işlemlerinin azlığı, `imbalanced-learn` kütüphanesinin SMOTE tekniği ile sadece eğitim verisi üzerinde çoğaltılarak sınıf dengesizliği giderilmiştir.
- Dengelenmiş veriler üzerinde yüksek performanslı `XGBClassifier` algoritması eğittim.
- Modelin başarısı, özellikle sahte işlemleri kaçırmama metrikleri olan Recall ve F1-Score üzerinden analiz edilmiş ve başarılı sonuçlar elde edilmiştir.

## Kullanım Talimatları (Kurulum ve Çalıştırma)
*1. **Gerekli Kütüphanelerin Kurulumu:**
   Proje dizininde terminali açın ve aşağıdaki komutu çalıştırın:
   pip install pandas scikit-learn xgboost imbalanced-learn streamlit
*2. **Veri Setinin Eklenmesi:**
   Kaggle üzerinden Credit Card Fraud Detection veri setini indirin ve içinden çıkan creditcard.csv dosyasını projenin ana klasörüne yerleştirin
*3. **Modelin Eğitilmesi:**
   Terminal üzerinden eğitim scriptini çalıştırarak modeli eğitin ve .pkl dosyasının oluşmasını sağlayın:
   python train_model.py
*4. **Web Uygulamasının Başlatılması:**
   Model eğitildikten sonra arayüzü ayağa kaldırmak için şu komutu kullanın:
   streamlit run app.py
