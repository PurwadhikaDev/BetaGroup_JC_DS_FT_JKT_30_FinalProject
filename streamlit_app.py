import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# ============================================================
# 🧭 PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Telco Customer Churn Prediction App",
    page_icon="📉",
    layout="wide"
)

# ============================================================
# 🧱 CUSTOM TRANSFORMERS (required for model loading)
# ============================================================
class NoOutlier(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): 
        return self
    def transform(self, X):    
        return X

class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5): 
        self.factor = factor
    def fit(self, X, y=None):
        X = np.asarray(X, float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self
    def transform(self, X): 
        return np.clip(np.asarray(X, float), self.lower_, self.upper_)

class QuantileWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99): 
        self.lower = lower
        self.upper = upper
    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.low_ = np.nanpercentile(X, self.lower * 100, axis=0)
        self.high_ = np.nanpercentile(X, self.upper * 100, axis=0)
        return self
    def transform(self, X): 
        return np.clip(np.asarray(X, float), self.low_, self.high_)

def impute_totalcharges(X):
    X = X.copy()
    mask = (X['totalcharges'].isna()) & (X['tenure'] == 0)
    X.loc[mask, 'totalcharges'] = X.loc[mask, 'monthlycharges']
    return X

# ============================================================
# 📦 LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load("churn_model.joblib")

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ============================================================
# 🧾 HEADER SECTION
# ============================================================
st.markdown("""
# 📉 **Telco Customer Churn Prediction App**
UI ini membaca struktur pipeline dari file model **`churn_model.joblib`** terbaru Anda —
nama kolom, urutan, dan kategori valid diambil langsung dari model.
""")

# ============================================================
# 🗂️ DEFINE TABS
# ============================================================
tab1, tab2 = st.tabs(["📊 Input & Prediksi", "ℹ️ Informasi Fitur & Model"])

# ============================================================
# 📊 TAB 1 – INPUT & PREDICTION
# ============================================================
with tab1:
    st.subheader("🧍 Input Data Pelanggan")
    st.write("Isi informasi pelanggan di bawah ini untuk memprediksi kemungkinan churn.")

    # -------------------------------
    # SIDEBAR-LIKE LAYOUT (COLUMNS)
    # -------------------------------
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        seniorcitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", 0, 100, 12)

    with col2:
        phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
        # Conditional logic for multiple lines
        if phoneservice == "No":
            multiplelines = "No phone service"
            st.info("Multiple Lines automatically set to 'No phone service'")
        else:
            multiplelines = st.selectbox("Multiple Lines", ["No", "Yes"])

        internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        if internetservice == "No":
            onlinesecurity = onlinebackup = deviceprotection = techsupport = streamingtv = streamingmovies = "No internet service"
            st.info("Internet-related services automatically set to 'No internet service'")
        else:
            onlinesecurity = st.selectbox("Online Security", ["Yes", "No"])
            onlinebackup = st.selectbox("Online Backup", ["Yes", "No"])
            deviceprotection = st.selectbox("Device Protection", ["Yes", "No"])
            techsupport = st.selectbox("Tech Support", ["Yes", "No"])
            streamingtv = st.selectbox("Streaming TV", ["Yes", "No"])
            streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No"])

    # -------------------------------
    # CONTRACT, BILLING, PAYMENT
    # -------------------------------
    st.markdown("### 💳 Contract, Billing & Payment Details")

    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperlessbilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    paymentmethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    monthlycharges = st.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0,
        help="Total monthly amount billed to the customer."
    )
    totalcharges = st.number_input(
        "Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0,
        help="Cumulative amount billed for all months of tenure."
    )

    # -------------------------------
    # PREPARE DATA FOR PREDICTION
    # -------------------------------
    input_data = pd.DataFrame({
        'gender': [gender],
        'seniorcitizen': [seniorcitizen],
        'partner': [partner],
        'dependents': [dependents],
        'tenure': [tenure],
        'phoneservice': [phoneservice],
        'multiplelines': [multiplelines],
        'internetservice': [internetservice],
        'onlinesecurity': [onlinesecurity],
        'onlinebackup': [onlinebackup],
        'deviceprotection': [deviceprotection],
        'techsupport': [techsupport],
        'streamingtv': [streamingtv],
        'streamingmovies': [streamingmovies],
        'contract': [contract],
        'paperlessbilling': [paperlessbilling],
        'paymentmethod': [paymentmethod],
        'monthlycharges': [monthlycharges],
        'totalcharges': [totalcharges]
    })

    # -------------------------------
    # RUN PREDICTION
    # -------------------------------
    if st.button("🔍 Predict Churn", type="primary"):
        try:
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]

            st.divider()
            st.subheader("🎯 Prediction Result")

            if prediction[0] in [1, "Yes"]:
                st.error(f"🚨 Pelanggan **berpotensi churn** (Probabilitas: {probability:.2%})")
            else:
                st.success(f"✅ Pelanggan **tidak berpotensi churn** (Probabilitas: {probability:.2%})")

            st.progress(probability)
            st.metric("Churn Probability", f"{probability*100:.2f}%")

            with st.expander("📋 View Input Data Summary"):
                st.dataframe(input_data, use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# ============================================================
# ℹ️ TAB 2 – FEATURE INFORMATION & MODEL DETAILS
# ============================================================
with tab2:
    st.subheader("📘 Informasi Fitur")
    st.write("Berikut merupakan daftar fitur yang digunakan dalam model prediksi churn:")

    # --- Create DataFrame from your provided dictionary ---
    feature_info = pd.DataFrame({
        "Fitur": [
            "CustomerID", "Gender", "Senior Citizen", "Partner", "Dependents",
            "Tenure in Months", "Phone Service", "Multiple Lines", "Internet Service",
            "Online Security", "Online Backup", "Device Protection Plan", "Premium Tech Support",
            "Streaming TV", "Streaming Movies", "Contract", "Paperless Billing",
            "Payment Method", "Monthly Charge", "Total Charges", "Churn Label"
        ],
        "Tipe": [
            "Teks", "Kategorikal", "Kategorikal", "Kategorikal", "Kategorikal",
            "Numerik", "Kategorikal", "Kategorikal", "Kategorikal",
            "Kategorikal", "Kategorikal", "Kategorikal", "Kategorikal",
            "Kategorikal", "Kategorikal", "Kategorikal", "Kategorikal",
            "Kategorikal", "Numerik", "Numerik", "Kategorikal"
        ],
        "Kategori/Contoh Nilai": [
            "3668-QPYBK, 9237-HQITU ...", "Male, Female", "0, 1", "Yes, No", "Yes, No",
            "0 – 72", "Yes, No", "Yes, No, No phone service", "DSL, Fiber optic, No",
            "Yes, No, No internet service", "Yes, No, No internet service", "Yes, No, No internet service",
            "Yes, No, No internet service", "Yes, No, No internet service", "Yes, No, No internet service",
            "Month-to-month, One year, Two year", "Yes, No",
            "Mailed check, Electronic check, Bank transfer, Credit card",
            "29.85 – 105.65 ...", "100.5 – 685.9 ...", "Yes, No"
        ],
        "Deskripsi": [
            "ID unik untuk setiap pelanggan.",
            "Jenis kelamin pelanggan.",
            "Apakah pelanggan berusia 65 tahun atau lebih.",
            "Apakah pelanggan memiliki pasangan.",
            "Apakah pelanggan memiliki tanggungan (anak/orang tua).",
            "Lama pelanggan berlangganan (dalam bulan).",
            "Apakah pelanggan memiliki layanan telepon rumah.",
            "Apakah pelanggan memiliki multiple line.",
            "Jenis layanan internet yang digunakan.",
            "Langganan layanan keamanan online.",
            "Langganan layanan backup online.",
            "Langganan perlindungan perangkat.",
            "Langganan dukungan teknis premium.",
            "Menggunakan internet untuk streaming TV.",
            "Menggunakan internet untuk streaming film.",
            "Jenis kontrak yang dimiliki pelanggan.",
            "Apakah menggunakan paperless billing.",
            "Metode pembayaran tagihan pelanggan.",
            "Tagihan bulanan pelanggan saat ini.",
            "Total biaya yang telah dibayar pelanggan hingga saat ini.",
            "Label churn (Yes = berhenti, No = tetap)."
        ]
    })

    # --- Display table with full-width container ---
    st.dataframe(feature_info, use_container_width=True, hide_index=True)

    # ------------------------------------------------
    # ℹ️ APPLICATION INFORMATION SECTION
    # ------------------------------------------------
    st.markdown("### ℹ️ **Informasi Aplikasi**")
    st.markdown("""
    - Aplikasi ini menggunakan **scikit-learn Pipeline** yang disimpan dalam file **`churn_model.joblib`**  
    untuk melakukan prediksi churn pelanggan.  
    - Beberapa *custom transformer* (`NoOutlier`, `IQRClipper`, `QuantileWinsorizer`) disertakan di kode ini  
    **bukan karena digunakan dalam model akhir**, melainkan agar Streamlit dapat memuat pipeline  
    tanpa error (sebagai *placeholder class* dari proses benchmarking sebelumnya).   
    - Struktur kolom input pada UI disusun **agar sesuai dengan urutan dan nama fitur saat training**,  
    sehingga mencegah error akibat ketidaksesuaian kolom (*mismatch*).  
    - Fitur-fitur yang bergantung pada layanan (seperti *Multiple Lines* dan *Internet-related services*)  
    memiliki **logika otomatis** untuk menyesuaikan nilai yang relevan berdasarkan input pengguna.  
    """)
