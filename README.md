## Project Overview

Industri telekomunikasi menghadapi persaingan yang sangat ketat, di mana pelanggan dengan mudah dapat berpindah ke penyedia layanan lain. Dalam bisnis berbasis langganan seperti telekomunikasi, **customer churn** (hilangnya pelanggan) menjadi ancaman utama bagi profitabilitas perusahaan.

Berbagai studi menunjukkan bahwa **mempertahankan pelanggan 5–7 kali lebih murah** dibandingkan memperoleh pelanggan baru. Faktor umum yang memengaruhi churn meliputi harga, kualitas layanan pelanggan, masalah penagihan, keandalan jaringan, serta kurangnya personalisasi.  
Dengan memanfaatkan **machine learning**, perusahaan telekomunikasi dapat **memprediksi pelanggan yang berisiko churn** dan merancang **strategi retensi** seperti program loyalitas, diskon, atau perpanjangan kontrak.

---

## Problem Statement

Tingkat **customer churn** perusahaan terus meningkat, sementara strategi retensi yang ada masih bersifat reaktif.  
Meskipun perusahaan telah memiliki data pelanggan yang lengkap, **belum ada mekanisme prediktif** yang dapat mengidentifikasi pelanggan berisiko tinggi secara akurat. Akibatnya, kampanye pemasaran menjadi kurang efisien dan potensi kehilangan pendapatan meningkat.

**Rumusan Masalah:**  
Bagaimana perusahaan dapat memprediksi pelanggan dengan risiko churn tinggi berdasarkan pola penggunaan layanan, jenis kontrak, dan karakteristik demografis agar dapat melakukan tindakan preventif untuk mengurangi churn?

---

## Project Goals

- Mengembangkan **model prediksi churn** yang dapat diintegrasikan ke dalam sistem **CRM** atau **Business Intelligence (BI)** untuk membantu tim marketing dalam menurunkan churn.  
- **Target Model:** F2-Score ≥ 70%, dengan fokus utama pada **recall** (mendeteksi pelanggan yang benar-benar churn).  
- **Target Bisnis:** Menurunkan churn rate hingga 10%, mengoptimalkan pengeluaran pemasaran, serta meningkatkan loyalitas pelanggan melalui tindakan proaktif.

---

## Analytic Approach

1. **Data Preparation** – pembersihan data dari missing values, outlier, dan duplikasi.  
2. **Exploratory Data Analysis (EDA)** – memahami demografi pelanggan, pola penggunaan layanan, dan hubungannya dengan churn.  
3. **Preprocessing** – encoding variabel kategorikal dan scaling pada fitur numerik.  
4. **Feature Selection** – mengidentifikasi fitur paling berpengaruh seperti tenure, jenis kontrak, dan metode pembayaran.  
5. **Model Development** – membandingkan performa antara **Linear Distance-Based Models**, **Tree-Based Models**, dan **Boosting Models**.  
6. **Evaluation** – menggunakan metrik **F2-Score** dan **Recall** untuk memprioritaskan deteksi pelanggan berisiko.  
7. **Interpretation** – menganalisis fitur penting dan menerjemahkan hasil model menjadi wawasan bisnis.  
8. **Deployment** – mengintegrasikan model ke dalam **Streamlit web-based tool** untuk prediksi churn secara real-time.

---

## Metric Evaluation

| | | Predicted | |
|---|---|---|---|
| | | No Churn (0) | Churn (1) |
| Actual | No Churn (0) | True Negative (TN) | False Positive (FP) |
|  | Churn (1) | False Negative (FN) | True Positive (TP) |

- **False Positive (Type I Error):** Predicts churn incorrectly, leading to unnecessary retention costs.  
- **False Negative (Type II Error):** Misses actual churners, resulting in direct revenue loss.

To minimize missed churners, the project focuses on Recall and uses the F2-Score, which gives Recall twice the weight of Precision:
F2 = (1 + 2^2) * (Precision * Recall) / ((2^2 * Precision) + Recall)

---

## Key Metrics

| Metrik | Rumus | Fokus |
|---------|--------|--------|
| **Precision** | TP / (TP + FP) | Menghindari pemborosan sumber daya pada pelanggan yang sebenarnya tidak churn |
| **Recall** | TP / (TP + FN) | Menangkap sebanyak mungkin pelanggan yang benar-benar churn |
| **F2-Score** | Rata-rata harmonik yang menekankan Recall | Menyeimbangkan precision dan recall dengan prioritas pada recall |

**Tujuan:** Mencapai nilai **Recall** dan **F2-Score ≥ 0.70**, untuk memastikan model mampu mendeteksi sebagian besar pelanggan yang berpotensi churn dengan tingkat ketepatan yang tetap dapat ditindaklanjuti.


## Data Sources and Description

### 1. Sumber Data

Dataset yang digunakan dalam proyek ini berasal dari **Telco Customer Churn Dataset** yang disediakan oleh **IBM** dan tersedia secara publik di [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).

Dataset ini berisi **7.043 baris data pelanggan** dan **21 atribut**, mencakup informasi demografi, layanan yang digunakan, jenis kontrak, metode pembayaran, serta status churn (apakah pelanggan berhenti berlangganan atau tidak).

---

### 2. Data Dictionary

| **Column Name** | **Description** | **Data Type** | **Unique Values (sample)** |
|-----------------|-----------------|----------------|----------------------------|
| `customerID` | ID unik untuk setiap pelanggan. | Text | 3668-QPYBK, 9237-HQITU, 9305-CDSKC |
| `gender` | Jenis kelamin pelanggan. | Categorical | Male, Female |
| `SeniorCitizen` | Menunjukkan apakah pelanggan berusia 65 tahun ke atas (0 = No, 1 = Yes). | Categorical | 0, 1 |
| `Partner` | Apakah pelanggan memiliki pasangan atau tidak. | Categorical | Yes, No |
| `Dependents` | Apakah pelanggan memiliki tanggungan (anak/orang tua). | Categorical | Yes, No |
| `tenure` | Lama berlangganan dalam bulan. | Numerical | 0–72 |
| `PhoneService` | Apakah pelanggan memiliki layanan telepon. | Categorical | Yes, No |
| `MultipleLines` | Apakah pelanggan memiliki lebih dari satu saluran telepon. | Categorical | Yes, No, No phone service |
| `InternetService` | Jenis layanan internet yang digunakan pelanggan. | Categorical | DSL, Fiber optic, No |
| `OnlineSecurity` | Berlangganan layanan keamanan online. | Categorical | Yes, No, No internet service |
| `OnlineBackup` | Berlangganan layanan backup online. | Categorical | Yes, No, No internet service |
| `DeviceProtection` | Berlangganan perlindungan perangkat. | Categorical | Yes, No, No internet service |
| `TechSupport` | Berlangganan dukungan teknis premium. | Categorical | Yes, No, No internet service |
| `StreamingTV` | Menggunakan layanan streaming TV. | Categorical | Yes, No, No internet service |
| `StreamingMovies` | Menggunakan layanan streaming film. | Categorical | Yes, No, No internet service |
| `Contract` | Jenis kontrak pelanggan. | Categorical | Month-to-month, One year, Two year |
| `PaperlessBilling` | Apakah pelanggan menggunakan sistem tagihan tanpa kertas. | Categorical | Yes, No |
| `PaymentMethod` | Metode pembayaran tagihan. | Categorical | Mailed check, Electronic check, Bank transfer, Credit card |
| `MonthlyCharges` | Biaya bulanan pelanggan. | Numerical | 29.85, 56.95, 105.65 |
| `TotalCharges` | Total biaya yang telah ditagihkan. | Numerical | 100.5, 245.3, 685.9 |
| `Churn` | Status churn pelanggan (apakah berhenti berlangganan). | Categorical | Yes, No |

---

### 3. Data Type Summary

| **Tipe Data** | **Jumlah Kolom** | **Contoh Kolom** |
|----------------|------------------|------------------|
| **Numerical** | 2 | `tenure`, `MonthlyCharges` |
| **Categorical** | 18 | `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `TotalCharges`, `Churn` |
| **Text (Identifier)** | 1 | `customerID` |
| **Total Features** | **21** | - |


## Technologies Used

Proyek ini dikembangkan menggunakan bahasa pemrograman **Python 3.12** serta berbagai library dan tools pendukung untuk proses analisis data, pengembangan model machine learning, dan deployment web-based tool menggunakan Streamlit.

### 1. Bahasa Pemrograman
- **Python 3.12**

### 2. Data Analysis & Visualization
- **pandas** – untuk manipulasi dan analisis data tabular  
- **numpy** – untuk perhitungan numerik  
- **matplotlib**, **seaborn** – untuk visualisasi data eksploratori  
- **Tableau** – untuk pembuatan dashboard interaktif dan laporan visual

### 3. Machine Learning & Modeling
- **scikit-learn** – untuk preprocessing, training, dan evaluasi model  
- **joblib** – untuk serialisasi (menyimpan dan memuat model)  
- **imbalanced-learn (imblearn)** – untuk menangani ketidakseimbangan data menggunakan teknik resampling seperti SMOTE  
- **xgboost**, **lightgbm** – untuk model Boosting berbasis gradient

### 4. Deployment & Web Framework
- **Streamlit** – untuk membangun dan menampilkan web-based prediction tool  
- **Git & GitHub** – untuk version control dan kolaborasi  
- **Streamlit Cloud** – untuk deployment dan hosting web tool secara online

### 5. Environment & Utilities
- **Jupyter Notebook / VS Code** – untuk eksplorasi dan pengembangan  
- **Google Colab** – untuk eksperimen dan pelatihan model di cloud  

## Project Structure

Struktur proyek ini dirancang agar mudah untuk dipahami, direproduksi, dan dikembangkan.  
Setiap folder memiliki fungsi spesifik untuk memisahkan tahap analisis data, pelatihan model, dan deployment Streamlit web-based tool.


```
telco-churn-app/
│
├── data/ # Berisi dataset mentah dan hasil preprocessing
│ └── customerchurn.csv
│
├── models/ # Berisi model yang telah dilatih dan diserialisasi
│ └── churn_model.joblib
│
├── notebooks/ # Notebook analisis dan pengembangan model
│ ├── finalproject.ipynb
│
├── references/ # Dokumen pendukung seperti sumber literatur, business context, dan catatan analisis
│
├── reports/ # Visualisasi hasil dan dashboard (mis. Tableau)
│ └── tableau_dashboard.twbx
│
├── streamlit_app.py # Source code utama untuk web-based churn prediction tool
├── requirements.txt # Daftar library dan dependencies proyek
├── README.md # Dokumentasi utama proyek
```

### Deskripsi Singkat Folder

| Folder | Deskripsi |
|--------|------------|
| **data/** | Berisi dataset mentah dan hasil preprocessing yang digunakan selama analisis dan pelatihan model. |
| **models/** | Berisi file model akhir (.joblib) yang digunakan oleh Streamlit untuk prediksi. |
| **notebooks/** | Menyimpan Jupyter Notebook untuk eksplorasi data (EDA), feature engineering, model training, dan evaluasi. |
| **references/** | Kumpulan sumber literatur, artikel, dan catatan referensi bisnis terkait churn analysis. |
| **reports/** | Menyimpan hasil visualisasi dan dashboard (termasuk laporan Tableau). |
| **streamlit_app.py** | Script utama yang menjalankan web-based prediction tool menggunakan Streamlit. |
| **requirements.txt** | Menyimpan daftar dependencies yang diperlukan untuk menjalankan proyek. |
| **README.md** | Dokumentasi utama proyek yang menjelaskan konteks, proses, dan hasil analisis. |

## Contact

**Co-Author:** Alief Dharmawan 
**Email:** aliefrnd@gmail.com
**LinkedIn:** [(https://www.linkedin.com/in/alief-dharmawan/)](https://www.linkedin.com/in/alief-dharmawan/)
**GitHub:** [[https://github.com/AliefRND](https://github.com/AliefRND))

**Co-Author:** Senoji Wicaksono 
**Email:** sena.aji30@gmail.com
**LinkedIn:** [(https://www.linkedin.com/in/senoaji-wicaksono-2871b416a/details/education/)](https://www.linkedin.com/in/senoaji-wicaksono-2871b416a/details/education/)
**GitHub:** [[https://github.com/senoaw017](https://github.com/senoaw017))

**Co-Author:** Sheyla Annisyah
**Email:** sheylaannisyah@gmail.com
**LinkedIn:** [(https://www.linkedin.com/in/sheylaannisyah/)](https://www.linkedin.com/in/sheylaannisyah/)
**GitHub:** [[https://github.com/SheylaAnn](https://github.com/SheylaAnn))


Proyek ini dikembangkan sebagai bagian dari **Final Project Data Science Bootcamp**, dengan fokus pada penerapan **machine learning untuk prediksi customer churn** di industri telekomunikasi.  
Seluruh pipeline analisis dan model dikembangkan menggunakan **Python 3.12**, **scikit-learn**, dan **Streamlit** untuk deployment berbasis web.

