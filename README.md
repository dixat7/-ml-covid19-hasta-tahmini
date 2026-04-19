# COVID-19 Survival Predictor: Machine Learning Classification 🦠🤖

Bu proje, Kaggle COVID-19 küresel veri setini kullanarak, bir vaka kaydında **ölüm gerçekleşip gerçekleşmediğini** tahmin eden uçtan uca bir Makine Öğrenmesi (Machine Learning) sınıflandırma modelidir.

Veri setindeki gözlemler analiz edilmiş, özellik mühendisliği (Feature Engineering) uygulanmış ve 4 farklı sınıflandırma algoritması eğitilerek performansları karşılaştırılmıştır.

## 🛠️ Kullanılan Teknolojiler
* **Dil:** Python
* **Veri İşleme:** Pandas, NumPy
* **Makine Öğrenmesi:** Scikit-Learn
* **Görselleştirme:** Matplotlib, Seaborn

## ⚙️ Proje Adımları
1. **Keşifsel Veri Analizi (EDA) & Temizlik:** * Eksik veriler (NaN) tespit edilip dolduruldu.
   * `Confirmed` (Vaka Sayısı) kolonundaki aşırı uç değerler (%99 persentil) filtrelenerek modeller outlier etkisinden korundu.
2. **Feature Engineering:** Zaman damgalarından "Ay" bilgisi çekilerek mevsimsel etkinin yakalanması sağlandı. Hedef değişken (Target) olarak `Deaths` kolonundan `Has_Deaths` (0 ve 1) adında ikili sınıflandırma kolonu üretildi. Veri sızıntısını (Data Leakage) önlemek için orijinal ölüm sayıları modelden çıkarıldı.
3. **Veri Ölçeklendirme:** Uzaklık tabanlı algoritmalar (Örn: KNN) için özellikler `StandardScaler` ile standardize edildi.
4. **Model Eğitimi:** Veriler %80 Eğitim, %20 Test olarak ayrıldı.

## 📊 Model Performansları ve Karşılaştırma

| Model Türü | Accuracy (Doğruluk) | Precision (Kesinlik) | Recall (Duyarlılık) |
| :--- | :---: | :---: | :---: |
| **Decision Tree** | %99.97 | %99.96 | %99.93 |
| **Random Forest** | %99.95 | %99.89 | %99.93 |
| **KNN (K-Nearest Neighbors)**| **%96.32** | **%92.34** | **%95.07** |
| **Logistic Regression** | %82.52 | %69.76 | %54.04 |

### 💡 Analiz ve Çıkarımlar
* **Ağaç Tabanlı Modeller (Overfitting):** Decision Tree ve Random Forest modelleri ~%99.9 doğruluk oranı ile veriyi ezberleme (overfitting) eğilimi göstermiştir.
* **Lojistik Regresyon:** Özellikle Recall (Duyarlılık) değerinin %54'te kalması, veriseti içerisindeki değişkenlerin doğrusal (linear) bir ilişkiye sahip olmadığını göstermektedir.
* **Projenin Kazananı (KNN):** Aşırı öğrenmeye düşmeden, %96.3 Doğruluk ve %95.0 Duyarlılık ile en dengeli ve güvenilir sonuçları K-Nearest Neighbors algoritması üretmiştir. Model, pozitif sınıfları (ölüm durumunu) yüksek oranda doğru yakalamıştır.

## 🚀 Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. Repoyu bilgisayarınıza klonlayın:
   ```bash
   git clone [https://github.com/KULLANICI_ADIN/covid-survival-predictor.git](https://github.com/KULLANICI_ADIN/covid-survival-predictor.git)
