#  Zomato Bengaluru Restaurant Analysis & Success Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Analisis komprehensif data restoran Zomato di Bengaluru untuk mengekstrak wawasan strategis bagi calon pemilik restoran, dilengkapi dengan model prediktif kesuksesan restoran menggunakan Machine Learning.**

---

##  Daftar Isi

- [Ringkasan Proyek](#ringkasan-proyek)
- [Problem Statement](#problem-statement)
- [Metodologi](#metodologi)
- [Key Findings](#key-findings)
- [Tech Stack](#️tech-stack)
- [Installation & Setup](#installation--setup)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Business Insights](#business-insights)
- [Future Improvements](#future-improvements)

---

## Ringkasan Proyek

Proyek ini menganalisis **51.000+ data restoran** dari platform Zomato di kota Bengaluru, India, untuk membantu **calon entrepreneur** membuat keputusan bisnis berbasis data. Analisis mencakup:

- **Exploratory Data Analysis (EDA)** — Distribusi pasar, tren harga, dan sentimen pelanggan
- **Text Analytics & Sentiment Analysis** — Ekstraksi insight dari 21.000+ review pelanggan
- **Market Segmentation** — K-Means Clustering untuk profiling wilayah Bengaluru
- **Predictive Modeling** — Klasifikasi "High Performer Restaurant" dengan **ROC-AUC 0.9996** (Random Forest)

**Business Value:**
> Mengubah data mentah menjadi **strategic recommendations** yang actionable untuk 3 skenario bisnis: Budget Terbatas, Modal Menengah, dan Premium Play.

---

## Problem Statement

**Tantangan Bisnis:**
Membuka restoran baru di Bengaluru adalah investasi berisiko tinggi dengan tingkat kegagalan yang signifikan. Calon pemilik restoran menghadapi ketidakpastian dalam:

1. **Pemilihan lokasi** — Wilayah mana yang paling profitable?
2. **Strategi penetapan harga** — Berapa harga optimal untuk maksimalkan revenue?
3. **Jenis masakan** — Cuisine apa yang paling diterima di lokasi tertentu?
4. **Adopsi fitur digital** — Apakah Online Order dan Book Table benar-benar meningkatkan popularitas?
5. **Waralaba vs Independen** — Apakah better memulai sebagai chain atau independent restaurant?

**Solusi:**
Membangun **decision support system berbasis data** yang memprediksi probabilitas kesuksesan restoran berdasarkan karakteristik bisnis yang dapat dikontrol sejak awal.

---

## Metodologi

### 1. Data Loading & Cleaning
- **Dataset:** 51.000+ baris data restoran dari Zomato Bengaluru
- **Pembersihan:** Handling missing values, duplicate removal, encoding fixes
- **Hasil:** 40.650 baris data siap analisis (80% retention rate)

### 2. Feature Engineering
Fitur baru yang dikonstruksi:
- `rest_type_main` — Tipe restoran utama (extracted dari multi-value field)
- `cuisine_main` — Masakan utama (primary cuisine identity)
- `is_chain` — Flag waralaba (1) vs independen (0) berdasarkan frekuensi nama
- `dish_count` — Jumlah menu populer sebagai proxy popularitas
- `sentiment_score` — Sentiment polarity dari review (-1 hingga +1)
- `complain_price` & `praise_taste` — Keyword flags untuk analisis

### 3. Text Analytics & Sentiment Analysis
- **Text Cleaning:** Regex-based cleaning + `ftfy` untuk Unicode fixes
- **Sentiment Analysis:** TextBlob untuk polarity scoring
- **Keyword Extraction:** Identification of complaint/praise patterns
- **WordCloud:** Visual representation of review themes per location

### 4. Exploratory Data Analysis (EDA)
**Visualisasi yang dibuat:**
- Distribusi tipe restoran (countplot)
- Korelasi harga vs sentimen (regplot dengan trendline)
- Sentiment per wilayah (barplot horizontal)
- Heatmap rating per masakan di setiap lokasi
- Chain vs Independent performance comparison (dual-axis plot)
- Digital features impact analysis (boxplot)

### 5. Market Segmentation (K-Means Clustering)
- **Algoritma:** K-Means dengan K=5 (optimal via Elbow Method)
- **Features:** Location, rating, cost, cuisine distribution (one-hot encoded)
- **Dimensionality Reduction:** PCA untuk visualisasi 2D
- **Output:** 5 cluster wilayah dengan profil strategis berbeda

**Cluster Profiles:**
| Cluster | Nama | Rating | Harga | Karakteristik |
|---------|------|--------|-------|---------------|
| 0 | Premium Enclave | 4.13 | Rp 1.304 | Lavelle Road — ultra-premium |
| 1 | Mass Market | 3.58 | Rp 539 | 43 wilayah — volume tinggi, price-sensitive |
| 2 | Underperforming | 3.64 | Rp 512 | RT Nagar — pasar belum mature |
| 3 | Premium Mid-Market | 3.67 | Rp 522 | Koramangala, Indiranagar — sweet spot |
| 4 | Upper Mid-Market | 3.82 | Rp 645 | Growth corridor |

### 6. Predictive Modeling

**Target Variable:**
```python
high_performer = (rating >= 4.0) & (votes >= 500)
```

**Models Trained:**
1. **Logistic Regression** (Baseline) — Accuracy: 90.84%, ROC-AUC: 0.9524
2. **Random Forest** (Best) — Accuracy: **99.48%**, ROC-AUC: **0.9996**
3. **XGBoost** — Accuracy: 98.39%, ROC-AUC: 0.9974

**Features Used (7 total):**
- `location`, `cuisine_main`, `online_order`, `book_table`, `cost`, `is_chain`, `rest_type_main`

**Evaluation Metrics:**
- ✅ ROC-AUC Curves comparison
- ✅ Confusion Matrices (3 models side-by-side)
- ✅ Feature Importance analysis (Logistic coef, RF gini, XGBoost gain)
- ✅ Comprehensive comparison table (Accuracy, Precision, Recall, F1-Score)

---

## Key Findings

### 1. Model Performance
> **Random Forest mencapai ROC-AUC 0.9996** — model dapat memprediksi kesuksesan restoran dengan akurasi sangat tinggi berdasarkan fitur bisnis yang tersedia sejak awal.

### 2. Top Success Factors (Feature Importance)
**Ranking berdasarkan Random Forest:**

1. 🥇 **Location (Lokasi)** — Impact tertinggi (>40%)
2. 🥈 **Cost (Harga)** — Penetapan harga yang tepat sangat krusial
3. 🥉 **Cuisine Type** — Jenis masakan harus match dengan demografi lokasi
4. **Book Table** — Fitur reservasi meningkatkan votes hingga **5.5x**
5. **Is Chain** — Waralaba memiliki advantage brand awareness
6. **Online Order** — Penting tapi tidak se-impactful Book Table
7. **Restaurant Type** — Menentukan ekspektasi pelanggan

### 3. Korelasi Harga-Sentimen
**r = 0.149** (sangat lemah)
> Restoran mahal **tidak otomatis** mendapat review lebih positif. Kualitas pengalaman pelanggan jauh lebih menentukan dibanding strategi harga semata.

### 4. Chain vs Independent
| Metric | Independen | Waralaba | Gap |
|--------|-----------|----------|-----|
| Rating | 3.54 | **3.71** | +4.8% |
| Votes | 71 | **282** | +298% |

> Waralaba mendominasi 92% pasar dan unggul signifikan dalam popularitas.

### 5. Digital Features Impact
| Fitur | Tidak Ada | Ada | Selisih |
|-------|-----------|-----|---------|
| Book Table → Votes | 193 | **1.077** | **+457%** |
| Book Table → Rating | 3.62 | **4.13** | +14% |
| Online Order → Votes | 325 | 323 | ~sama |

> **Book Table adalah game-changer** untuk Casual Dining & Fine Dining.

---

## Tech Stack

### Programming & Data Science
- **Python 3.8+**
- **Jupyter Notebook**
- **pandas** — Data manipulation
- **NumPy** — Numerical computing

### Machine Learning
- **scikit-learn** — Model training & evaluation
  - Logistic Regression
  - Random Forest Classifier
  - Label Encoding, Train-Test Split, Metrics
- **XGBoost** — Gradient boosting classifier

### Natural Language Processing
- **TextBlob** — Sentiment analysis
- **ftfy** — Unicode text cleaning
- **WordCloud** — Text visualization

### Data Visualization
- **matplotlib** — Base plotting
- **seaborn** — Statistical visualizations
  - countplot, regplot, heatmap, boxplot, barplot

### Clustering & Dimensionality Reduction
- **KMeans** — Market segmentation
- **PCA** — Principal Component Analysis
- **StandardScaler** — Feature scaling

---

## Installation & Setup

### Prerequisites
```bash
# Pastikan Python 3.8+ terinstall
python --version
```

### Clone Repository
```bash
git clone https://github.com/fazqin/Zomato_bengaluru_analysis.git
cd zomato_bengaluru_analysis
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Jupyter Notebook
```bash
jupyter notebook Zomato_Final.ipynb
```

**Atau buka dengan:**
- VS Code + Jupyter extension
- Google Colab (upload .ipynb)
- JupyterLab

---

## Project Structure

```
zomato-bengaluru-analysis/
│
├── Zomato_Final.ipynb              # Main analysis notebook
├── zomato.csv                       # Raw dataset (51k+ rows)
├── README.md                        # Project documentation (this file)
├── requirements.txt                 # Python dependencies
├── Zomato_Final_backup_*.ipynb     # Backup files (auto-generated)
│
└── fix_notebook_order.py           # Utility script (cell reordering)
```

---

## Model Performance

### Random Forest (Best Model)

```python
Accuracy:  99.48%
ROC-AUC:   0.9996
Precision: 0.9827
Recall:    0.9767
F1-Score:  0.9797
```

**Confusion Matrix:**
```
                     Predicted
                   Non-HP    HP
Actual  Non-HP     7050      8
        HP           25    1047
```

**Interpretation:**
- **False Positives:** 8 restoran (diprediksi HP tapi ternyata tidak) — 0.11%
- **False Negatives:** 25 restoran (diprediksi non-HP tapi ternyata HP) — 2.33%
- Model sangat akurat dengan **minimal error** pada kedua kelas

### Model Comparison

| Model | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
|-------|----------|---------|-----------|--------|----------|
| **Random Forest** | **0.9948** | **0.9996** | **0.9827** | **0.9767** | **0.9797** |
| XGBoost | 0.9839 | 0.9974 | 0.9253 | 0.9234 | 0.9244 |
| Logistic Regression | 0.9084 | 0.9524 | 0.6826 | 0.5504 | 0.6093 |

---

## Business Insights

### Strategic Recommendations

####  Skenario 1: Budget Terbatas (Modal < Rp 50 juta)
**Rekomendasi:**
- ✅ Lokasi: **Mass Market** (BTM, Whitefield, Marathahalli) — sewa murah
- ✅ Konsep: **Quick Bites** (Rp 200-400 untuk 2 orang)
- ✅ Masakan: **North Indian** atau **South Indian** (demand stabil)
- ✅ Prioritas: **Online Order** sejak hari pertama
- ❌ **JANGAN:** Coba masuk area premium — barrier to entry terlalu tinggi

####  Skenario 2: Modal Menengah (Rp 50-150 juta)
**Rekomendasi:**
- ✅ Lokasi: **Premium Mid-Market** (Koramangala 5th Block, HSR, Jayanagar)
- ✅ Konsep: **Cafe** atau **Casual Dining** dengan twist unik
- ✅ Harga: Rp 600-900 (sweet spot affordable-premium)
- ✅ Features: **Book Table + Online Order** (full digital adoption)
- ✅ Masakan: Eksperimen **fusion** atau niche (Mexican, Continental)
- ✅ Investment: Interior design & Instagram-worthy ambience

####  Skenario 3: Modal Besar (Rp 150 juta+)
**Rekomendasi:**
- ✅ Lokasi: **Premium Enclave** (Lavelle Road, MG Road)
- ✅ Konsep: **Fine Dining** atau **Lounge** dengan exceptional service
- ✅ Harga: Rp 1.200+ (justify dengan experience & quality)
- ✅ Book Table: **WAJIB** — target corporate dinners & celebrations
- ✅ Masakan: **Continental** atau **Fusion Premium**
- ⚠️ **WARNING:** Gagal di segment ini sangat mahal — hanya masuk jika yakin bisa deliver 4.5+ rating

### Key Takeaways untuk Entrepreneur

1.  **Lokasi adalah segalanya** — riset mendalam sebelum commit
2.  **Harga harus match dengan value proposition**, bukan asal mahal
3.  **Book Table > Online Order** untuk meningkatkan popularitas
4.  **Waralaba memiliki advantage**, independen harus punya diferensiasi kuat
5.  **Type & Cuisine** harus aligned dengan target market lokasi

---

## Future Improvements

### Model Enhancements
- [ ] **Hyperparameter Tuning** — GridSearchCV untuk optimize Random Forest & XGBoost
- [ ] **Cross-Validation** — K-Fold CV untuk validate model robustness
- [ ] **Ensemble Stacking** — Combine predictions dari multiple models
- [ ] **SMOTE** — Handle class imbalance dengan synthetic sampling
- [ ] **Time-Series Analysis** — Incorporate temporal trends (seasonality)

### Feature Engineering
- [ ] **Sentiment Score sebagai Feature** — Integrate NLP insights ke model
- [ ] **Topic Modeling** — LDA atau BERTopic untuk review themes
- [ ] **External Data** — Footfall traffic, nearby competitors count, parking availability
- [ ] **Distance to Landmarks** — Proximity to metro stations, malls, corporate hubs

### Deployment & Productionization
- [ ] **Streamlit Dashboard** — Interactive web app untuk business users
- [ ] **FastAPI Backend** — REST API untuk real-time predictions
- [ ] **Docker Containerization** — Portable deployment
- [ ] **Model Monitoring** — Track drift & performance degradation
- [ ] **A/B Testing Framework** — Validate recommendations in production

### Code Quality
- [ ] **Modular Code Structure** — `src/` folder dengan reusable functions
- [ ] **Unit Tests** — `pytest` untuk critical functions
- [ ] **CI/CD Pipeline** — Automated testing & deployment
- [ ] **Documentation** — Docstrings, type hints, API docs

---

## Acknowledgments

- **Dataset:** [Zomato Bangalore Restaurants](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants) from Kaggle
- **Inspiration:** Real-world challenges faced by restaurant entrepreneurs in Bengaluru
- **Libraries:** Thanks to the amazing open-source community for scikit-learn, pandas, and matplotlib
---

<div align="center">

** Star this repo jika bermanfaat!**

Made with and using Python & Jupyter

</div>
