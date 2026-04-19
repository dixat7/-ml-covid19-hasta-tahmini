import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# ==========================================
# 1. VERİ YÜKLEME VE EDA (Keşifsel Veri Analizi)
# ==========================================
print("--- Veri Yükleniyor ---")
# Dosya yolunu kendi sisteminize göre güncelleyebilirsiniz.
df = pd.read_csv("covid_19_data.csv")

print("\n--- Veri Setinin İlk 5 Satırı ---")
print(df.head())

print("\n--- Veri Seti Bilgisi (Info) ---")
df.info()

print("\n--- Eksik Değer Kontrolü ---")
print(df.isnull().sum())


# ==========================================
# 2. VERİ TEMİZLEME VE ÖN İŞLEME
# ==========================================
print("\n--- Veri Temizleme İşlemleri Başlıyor ---")

# a) Gereksiz sütunların düşürülmesi
# SNo (Sıra numarası) ve Last Update (Son güncelleme - tarihsel tutarlılık için ObservationDate yeterli) düşürülüyor.
df = df.drop(['SNo', 'Last Update'], axis=1)

# b) Eksik verilerin doldurulması
# Province/State sütunundaki eksik veriler (NaN) 'Unknown' (Bilinmiyor) olarak dolduruluyor.
df['Province/State'] = df['Province/State'].fillna('Unknown')

# c) Tarih verisinden anlamlı özellik üretme (Feature Engineering)
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
df['Month'] = df['ObservationDate'].dt.month # Sadece ayı çekiyoruz
df = df.drop(['ObservationDate'], axis=1)

# d) Sınıflandırma Problemi İçin Hedef Değişken (Target) Üretme
# Hedef: Ölüm sayısı 0'dan büyükse 1 (Ölüm Var), değilse 0 (Ölüm Yok)
df['Has_Deaths'] = df['Deaths'].apply(lambda x: 1 if x > 0 else 0)

# Artık hedef değişkenimizi oluşturduğumuz için 'Deaths' sütununu silebiliriz (Veri sızıntısını / Data Leakage'i önlemek için)
df = df.drop(['Deaths'], axis=1)

# e) Kategorik değişkenleri sayısal formata çevirme (Label Encoding)
le = LabelEncoder()
df['Country/Region'] = le.fit_transform(df['Country/Region'])
df['Province/State'] = le.fit_transform(df['Province/State'])

# f) Aykırı Değer (Outlier) Temizliği
# Aşırı yüksek Confirmed (Onaylanmış Vaka) sayılarını (%99. persentil üzeri) filtreleyerek modeli uç değerlerden koruyoruz.
q99 = df['Confirmed'].quantile(0.99)
df_cleaned = df[df['Confirmed'] <= q99]

# ==========================================
# 3. MODEL EĞİTİMİNE HAZIRLIK (Split & Scaling)
# ==========================================
# Bağımsız değişkenler (X) ve Bağımlı değişken (y)
X = df_cleaned.drop('Has_Deaths', axis=1)
y = df_cleaned['Has_Deaths']

# Veriyi %80 Eğitim (Train) - %20 Test (Test) olarak bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veri Ölçeklendirme (Standardization) - Özellikle KNN ve Lojistik Regresyon için çok önemlidir.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 4. MODEL EĞİTİMİ VE TEST İŞLEMLERİ
# ==========================================
# İstenen algoritmalar bir sözlük içerisinde tanımlanıyor
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN (K-Nearest Neighbors)": KNeighborsClassifier(n_neighbors=5)
}

# ==========================================
# 5. MODEL PERFORMANS DEĞERLENDİRMESİ
# ==========================================
for name, model in models.items():
    print(f"\n==========================================")
    print(f" {name} Modeli Performansı")
    print(f"==========================================")
    
    # 1. Modeli Eğit (Train)
    model.fit(X_train_scaled, y_train)
    
    # 2. Test verisi üzerinde tahmin yap (Predict)
    y_pred = model.predict(X_test_scaled)
    
    # 3. Metrikleri Hesapla
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy (Doğruluk) : {acc:.4f}")
    print(f"Precision (Kesinlik): {prec:.4f}")
    print(f"Recall (Duyarlılık) : {rec:.4f}")
    
    # 4. Confusion Matrix (Karmaşıklık Matrisi) Görselleştirme
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ölüm Yok (0)', 'Ölüm Var (1)'], 
                yticklabels=['Ölüm Yok (0)', 'Ölüm Var (1)'])
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Modelin Tahmini')
    plt.ylabel('Gerçek Durum')
    plt.tight_layout()
    plt.show() # VS Code üzerinde pop-up pencere açarak grafikleri sırayla gösterecektir.