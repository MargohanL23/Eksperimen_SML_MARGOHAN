"""
automate_MARGOHAN.py
----------------------------------------------------
Script otomatisasi preprocessing dan pelatihan model
untuk dataset German Credit Risk.

Dibuat berdasarkan eksperimen manual di notebook:
Eksperimen_MARGOHAN.ipynb
----------------------------------------------------
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ======================================================
# 1️⃣ Fungsi: Load Dataset
# ======================================================
def load_dataset(path):
    """Memuat dataset dari path CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di {path}")
    df = pd.read_csv(path)
    print(f"✅ Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


# ======================================================
# 2️⃣ Fungsi: Preprocessing Data
# ======================================================
def preprocess_data(df):
    """Melakukan pembersihan dan preprocessing awal pada data."""
    # Hapus kolom yang tidak relevan
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # Tangani missing values
    for col in ['Saving accounts', 'Checking account']:
        if col in df.columns:
            df[col].fillna('Unknown', inplace=True)

    print("✅ Missing values ditangani dan data dibersihkan.")
    return df


# ======================================================
# 3️⃣ Fungsi: Split Data
# ======================================================
def split_data(df):
    """Memisahkan data menjadi fitur (X) dan target (y)."""
    X = df.drop('Risk', axis=1)
    y = df['Risk'].map({'good': 0, 'bad': 1})  # konversi target ke numerik

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✅ Data berhasil di-split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ======================================================
# 4️⃣ Fungsi: Buat Pipeline
# ======================================================
def build_pipeline():
    """Membuat pipeline untuk preprocessing + model."""
    numerical_features = ['Age', 'Credit amount', 'Duration']
    categorical_features = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Model dasar
    model = RandomForestClassifier(random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    print("✅ Pipeline (Preprocessor + Model) berhasil dibuat.")
    return pipeline


# ======================================================
# 5️⃣ Fungsi: Latih dan Evaluasi Model
# ======================================================
def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test):
    """Melatih model dan menampilkan hasil evaluasi."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n📊 Evaluasi Model:")
    print(classification_report(y_test, y_pred))
    print(f"Akurasi: {accuracy_score(y_test, y_pred):.4f}")

    return pipeline


# ======================================================
# 6️⃣ Fungsi: Simpan Model
# ======================================================
def save_model(pipeline, save_path):
    """Menyimpan pipeline model ke file .pkl"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"✅ Model tersimpan di: {save_path}")


# ======================================================
# 7️⃣ MAIN EXECUTION
# ======================================================
if __name__ == "__main__":
    # Path dataset & output
    dataset_path = r"C:\Users\indah\OneDrive\Desktop\SMSML_MARGOHAN\Eksperimen_SML_MARGOHAN\namadataset_raw\german_credit_data.csv"
    model_output_path = r"C:\Users\indah\OneDrive\Desktop\SMSML_MARGOHAN\Eksperimen_SML_MARGOHAN\preprocessing\model.pkl"

    # Jalankan semua tahapan otomatis
    df = load_dataset(dataset_path)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    pipeline = build_pipeline()
    trained_pipeline = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)
    save_model(trained_pipeline, model_output_path)

    print("\n🎯 Semua proses selesai tanpa error. Model siap digunakan!")
