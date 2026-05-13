import streamlit as st
import pandas as pd
import joblib
import random
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Any

class FraudDetectionApp:
    """
    A Streamlit application class for the Credit Card Fraud Detection System.
    
    This class manages the user interface, data loading, model inference, 
    and result visualization using Streamlit.
    """

    def __init__(self, data_path: str = 'creditcard.csv', model_path: str = 'fraud_detection_model.pkl'):
        """
        Initializes the FraudDetectionApp with data and model paths.

        Args:
            data_path (str): Path to the credit card dataset.
            model_path (str): Path to the trained machine learning model.
        """
        self.data_path = data_path
        self.model_path = model_path
        
        # Initialize session state for query history
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []

    @staticmethod
    @st.cache_data
    def load_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads and preprocesses the credit card dataset. Results are cached by Streamlit.

        Args:
            data_path (str): Path to the dataset CSV file.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the raw dataframe 
            and the preprocessed dataframe (scaled Amount and Time).
        """
        df_raw = pd.read_csv(data_path)
        
        df_processed = df_raw.copy()
        scaler = StandardScaler()
        df_processed['Amount'] = scaler.fit_transform(df_processed['Amount'].values.reshape(-1, 1))
        df_processed['Time'] = scaler.fit_transform(df_processed['Time'].values.reshape(-1, 1))
        
        return df_raw, df_processed

    @staticmethod
    @st.cache_resource
    def load_model(model_path: str) -> Any:
        """
        Loads the trained machine learning model. Results are cached by Streamlit.

        Args:
            model_path (str): Path to the trained model PKL file.

        Returns:
            Any: The loaded machine learning model object (e.g., XGBClassifier).
        """
        return joblib.load(model_path)

    def render_sidebar(self) -> None:
        """
        Renders the sidebar containing system information and creates placeholders for query statistics.
        """
        st.sidebar.title("Kredi Kartı Dolandırıcılık Tespit Sistemi")
        st.sidebar.markdown(
            "Bu sistem, finansal işlem verilerini analiz ederek makine "
            "öğrenimi algoritmalarıyla anormallik tespiti yapmaktadır."
        )
        st.sidebar.markdown("---")
        
        st.sidebar.subheader("Sistem Özeti")
        
        # 1. Spesisifk yer tutucuları tanımlıyoruz
        col1, col2 = st.sidebar.columns(2)
        self.total_placeholder = col1.empty()
        self.fraud_placeholder = col2.empty()
        self.normal_placeholder = st.sidebar.empty()
        st.sidebar.markdown("---")
        
        # 2. Titremeyi (flickering) önlemek için ilk başta mevcut değerlerle dolduruyoruz
        self._fill_placeholders()

    def _fill_placeholders(self) -> None:
        """
        Calculates current metrics and writes them to placeholders.
        """
        total_queries = len(st.session_state.query_history)
        fraud_found = sum(1 for q in st.session_state.query_history if q['Model Tahmini'] == "Dolandırıcı (1)")
        normal_approved = sum(1 for q in st.session_state.query_history if q['Model Tahmini'] == "Normal (0)")
        
        self.total_placeholder.metric("Toplam Sorgu", total_queries)
        self.fraud_placeholder.metric("Sahte", fraud_found)
        self.normal_placeholder.metric("Onaylanan Normal İşlem", normal_approved)

    def update_sidebar_metrics(self) -> None:
        """
        Updates the sidebar metrics placeholders with the latest session state data.
        """
        self._fill_placeholders()

    def simulate_transaction(self, target_class: int, df_raw: pd.DataFrame, df_processed: pd.DataFrame, model: Any, tab_main: Any, tab_xai: Any) -> None:
        """
        Simulates a transaction prediction by randomly picking a sample from the dataset.

        Args:
            target_class (int): 0 for Normal, 1 for Fraudulent transaction.
            df_raw (pd.DataFrame): The raw dataframe containing original values.
            df_processed (pd.DataFrame): The preprocessed dataframe for model input.
            model (Any): The trained machine learning model.
            tab_main (Any): Streamlit tab object for the main analysis display.
            tab_xai (Any): Streamlit tab object for the XAI visualization display.
        """
        subset_indices = df_raw[df_raw['Class'] == target_class].index
        
        if len(subset_indices) == 0:
            with tab_main:
                st.warning("Bu sınıfa ait veri bulunamadı.")
            return
            
        random_index = random.choice(subset_indices)
        
        real_amount = df_raw.loc[random_index, 'Amount']
        real_time = df_raw.loc[random_index, 'Time']
        
        features_df = df_processed.drop('Class', axis=1).loc[[random_index]]
        
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        confidence_score = probabilities[prediction] * 100
        
        self._display_transaction_details(
            real_amount, real_time, features_df, df_raw.loc[[random_index]].drop('Class', axis=1), 
            prediction, confidence_score, tab_main
        )
        self._display_xai_chart(features_df, model, prediction, tab_xai)
        self._update_history(real_amount, real_time, target_class, prediction, confidence_score)

    def _display_transaction_details(self, amount: float, time: float, features_df: pd.DataFrame, 
                                     raw_features_df: pd.DataFrame, prediction: int, 
                                     confidence: float, tab: Any) -> None:
        """
        Displays transaction details and prediction results in the specified tab.

        Args:
            amount (float): Transaction amount.
            time (float): Transaction time.
            features_df (pd.DataFrame): Feature dataframe used for prediction.
            raw_features_df (pd.DataFrame): Raw feature dataframe for display.
            prediction (int): The model's predicted class.
            confidence (float): The model's confidence score percentage.
            tab (Any): The Streamlit tab to render the content in.
        """
        with tab:
            st.subheader("İşlem Detayları")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="İşlem Tutarı (Amount)", value=f"${amount:,.2f}")
            with col2:
                st.metric(label="İşlem Zamanı (Time)", value=f"{time:.0f}")
                
            with st.expander("Makine Öğrenimi Arka Plan Verilerini Göster (V1-V28)"):
                st.dataframe(raw_features_df, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Model Tahmini")
            
            if prediction == 0:
                st.success(f"✅ GÜVENLİ İŞLEM ONAYLANDI (Modelin Eminlik Oranı: %{confidence:.0f})")
            else:
                st.error(f"🚨 DİKKAT: DOLANDIRICILIK ŞÜPHESİ! (Modelin Eminlik Oranı: %{confidence:.0f})")
                
            st.progress(int(confidence))

    def _display_xai_chart(self, features_df: pd.DataFrame, model: Any, prediction: int, tab: Any) -> None:
        """
        Renders the Explainable AI (XAI) feature importance chart.

        Args:
            features_df (pd.DataFrame): Feature dataframe used for prediction.
            model (Any): The trained machine learning model.
            prediction (int): The model's predicted class.
            tab (Any): The Streamlit tab to render the chart in.
        """
        with tab:
            st.subheader("Modelin Karar Mekanizması")
            st.markdown("Modelin bu işlemi sınıflandırırken en çok dikkat ettiği 5 özellik:")
            
            if hasattr(model, 'feature_importances_'):
                import altair as alt
                importances = model.feature_importances_
                
                importance_df = pd.DataFrame({
                    'Özellik': features_df.columns,
                    'Önem Skoru': importances
                }).sort_values(by='Önem Skoru', ascending=False).head(5)
                
                bar_color = '#ff4b4b' if prediction == 1 else '#00cc96'
                
                chart = alt.Chart(importance_df).mark_bar().encode(
                    x='Önem Skoru:Q',
                    y=alt.Y('Özellik:N', sort='-x'),
                    color=alt.value(bar_color),
                    tooltip=['Özellik', 'Önem Skoru']
                ).properties(height=250)
                
                st.altair_chart(chart, use_container_width=True)

    def _update_history(self, amount: float, time: float, true_class: int, prediction: int, confidence: float) -> None:
        """
        Updates the session state query history with the latest transaction simulation.

        Args:
            amount (float): Transaction amount.
            time (float): Transaction time.
            true_class (int): The actual class of the simulated transaction.
            prediction (int): The predicted class by the model.
            confidence (float): The confidence score percentage.
        """
        prediction_text = "Dolandırıcı (1)" if prediction == 1 else "Normal (0)"
        true_class_text = "Dolandırıcı (1)" if true_class == 1 else "Normal (0)"
        
        st.session_state.query_history.append({
            "İşlem Tutarı": f"${amount:,.2f}",
            "Zaman": f"{time:.0f}",
            "Gerçek Sınıf": true_class_text,
            "Model Tahmini": prediction_text,
            "Güven Skoru": f"%{confidence:.0f}"
        })

    def run(self) -> None:
        """
        Main execution method for the Streamlit application.
        Sets up the page layout, handles data loading, and renders all UI components.
        """
        st.set_page_config(page_title="Dolandırıcılık Tespit Sistemi", layout="wide")
        
        self.render_sidebar()
        
        try:
            with st.spinner("Veri seti ve model yükleniyor..."):
                # Call static methods to utilize st.cache correctly
                df_raw, df_processed = self.load_data(self.data_path)
                model = self.load_model(self.model_path)
        except Exception as e:
            st.error(f"Veri veya model yüklenirken bir hata oluştu: {e}")
            st.info("Lütfen 'creditcard.csv' ve 'fraud_detection_model.pkl' dosyalarının uygulama ile aynı klasörde olduğundan emin olun.")
            return

        st.markdown("""
        <h1 style='text-align: center; color: #1E3A8A;'>Kredi Kartı Dolandırıcılık Tespit Merkezi</h1>
        """, unsafe_allow_html=True)
        
        tab_main, tab_history, tab_xai = st.tabs([
            "🔍 Anlık Analiz Paneli", 
            "📊 İşlem Geçmişi", 
            "🧠 Model Açıklanabilirliği (XAI)"
        ])
        
        with tab_main:
            st.markdown("### Bir İşlem Seçin")
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                btn_normal = st.button("✅ Güvenli İşlem Simüle Et", use_container_width=True)
            with col_btn2:
                btn_fraud = st.button("🚨 Şüpheli İşlem Simüle Et", use_container_width=True)
            st.markdown("---")
        
        if btn_normal:
            self.simulate_transaction(0, df_raw, df_processed, model, tab_main, tab_xai)
        elif btn_fraud:
            self.simulate_transaction(1, df_raw, df_processed, model, tab_main, tab_xai)
        else:
            with tab_main:
                st.info("👆 Lütfen analiz sürecini başlatmak için yukarıdaki butonlardan birini seçin.")
            with tab_xai:
                st.info("👈 Henüz bir işlem seçilmediği için açıklanabilirlik grafiği gösterilemiyor.")
                
        with tab_history:
            st.subheader("Geçmiş Sorgulamalar")
            if len(st.session_state.query_history) > 0:
                history_df = pd.DataFrame(st.session_state.query_history)
                st.dataframe(history_df, use_container_width=True)
            else:
                st.info("Henüz sorgu yapılmadı.")
                
        self.update_sidebar_metrics()

if __name__ == "__main__":
    app = FraudDetectionApp()
    app.run()
