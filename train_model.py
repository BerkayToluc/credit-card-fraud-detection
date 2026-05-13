import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import joblib
import os
from typing import Tuple

class FraudDetectionTrainer:
    """
    A class used to train an XGBoost model for credit card fraud detection.
    
    This class handles data loading, preprocessing (StandardScaler and SMOTE),
    model training, evaluation, and saving the trained model.
    
    Attributes:
        data_path (str): The path to the credit card dataset (CSV).
        model_filename (str): The filename for the saved model (PKL).
        model (XGBClassifier): The XGBoost model instance.
    """

    def __init__(self, data_path: str = 'creditcard.csv', model_filename: str = 'fraud_detection_model.pkl'):
        """
        Initializes the FraudDetectionTrainer with file paths.

        Args:
            data_path (str): Path to the input dataset. Defaults to 'creditcard.csv'.
            model_filename (str): Path to save the trained model. Defaults to 'fraud_detection_model.pkl'.
        """
        self.data_path = data_path
        self.model_filename = model_filename
        self.model = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset from the specified CSV file.

        Returns:
            pd.DataFrame: The loaded dataset.
            
        Raises:
            FileNotFoundError: If the dataset file does not exist.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Hata: '{self.data_path}' bulunamadı.")
        
        print("Veri seti yükleniyor...")
        return pd.read_csv(self.data_path)

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocesses the dataframe by scaling features and separating the target variable.

        Args:
            df (pd.DataFrame): The raw input dataframe.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: A tuple containing the feature matrix (X) 
            and the target vector (y).
        """
        print("Veri ön işleme: 'Amount' ve 'Time' sütunları standartlaştırılıyor...")
        scaler = StandardScaler()
        
        # Scale 'Amount' and 'Time' columns
        df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
        
        # Separate features (X) and target (y)
        x_features = df.drop('Class', axis=1)
        y_target = df['Class']
        
        return x_features, y_target

    def split_and_balance_data(self, x_features: pd.DataFrame, y_target: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits data into train and test sets, and applies SMOTE to balance the training data.

        Args:
            x_features (pd.DataFrame): The feature matrix.
            y_target (pd.Series): The target vector.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Balanced training features, 
            test features, balanced training targets, and test targets.
        """
        print("Veri seti eğitim ve test olarak bölünüyor...")
        x_train, x_test, y_train, y_test = train_test_split(
            x_features, y_target, test_size=0.2, random_state=42, stratify=y_target
        )
        
        print("Eğitim verisine SMOTE uygulanarak 0 ve 1 sınıfları eşitleniyor...")
        smote = SMOTE(random_state=42)
        x_train_balanced, y_train_balanced = smote.fit_resample(x_train, y_train)
        
        return x_train_balanced, x_test, y_train_balanced, y_test

    def train_model(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Trains the XGBoost classifier on the provided training data.

        Args:
            x_train (pd.DataFrame): The training feature matrix.
            y_train (pd.Series): The training target vector.
        """
        print("Model eğitiliyor (XGBoost)... Bu işlem biraz zaman alabilir.")
        self.model = XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss', 
            random_state=42, 
            n_jobs=-1
        )
        self.model.fit(x_train, y_train)

    def evaluate_model(self, x_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Evaluates the trained model using the test data and prints performance metrics.

        Args:
            x_test (pd.DataFrame): The test feature matrix.
            y_test (pd.Series): The test target vector.
            
        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi. Lütfen önce train_model() metodunu çağırın.")
            
        print("Test seti üzerinde tahmin yapılıyor...")
        y_pred = self.model.predict(x_test)
        
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

    def save_model(self) -> None:
        """
        Saves the trained model to disk as a PKL file.
        
        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi. Kaydedilecek model bulunamadı.")
            
        joblib.dump(self.model, self.model_filename)
        print(f"\nBaşarılı: Eğitilmiş model '{self.model_filename}' olarak kaydedildi!")

    def run_pipeline(self) -> None:
        """
        Executes the entire end-to-end training pipeline:
        loading, preprocessing, splitting, training, evaluating, and saving.
        """
        try:
            df_raw = self.load_data()
            x_features, y_target = self.preprocess_data(df_raw)
            x_train, x_test, y_train, y_test = self.split_and_balance_data(x_features, y_target)
            
            self.train_model(x_train, y_train)
            self.evaluate_model(x_test, y_test)
            self.save_model()
        except Exception as e:
            print(f"Boru hattı (pipeline) yürütülürken hata oluştu: {e}")

if __name__ == "__main__":
    trainer = FraudDetectionTrainer()
    trainer.run_pipeline()
