# Anomaly Detection in Financial Transactions — Teknik Rapor

**Ders:** İş Yeri Uygulaması (IYU 228) – Faz 2 · **Hazırlayan:** Uluhan Yusuf Dokay · **Teslim:** 01.07.2025

## 1. Amaç

Kredi kartı işlemleri içindeki dolandırıcılık (fraud) kayıtlarını, etiket bilgisi kullanmadan
eğitilen anomali tespit modelleriyle bulmak ve üç yöntemi karşılaştırmak.

## 2. Veri Seti

Kaynak: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Özellik | Değer |
|---|---|
| Satır | 284.807 işlem |
| Fraud | 492 (%0,17) |
| Sütunlar | `Time`, `V1`–`V28` (PCA), `Amount`, `Class` |

Fraud oranının çok düşük olması hem modelleme hem de metrik seçimi açısından temel zorluktur
(yalnızca doğruluk oranına bakmak yanıltıcıdır: hepsini "normal" diyen model %99,8 doğrulukta olur).

## 3. Yöntem

### 3.1 Ön işleme (`scripts/preprocessing.py`)
- Eksik değer kontrolü (veri setinde eksik değer yoktur).
- Sınıf dağılımı grafiği (`results/class_distribution.png`, log ölçek).
- %70 eğitim / %30 test, `stratify=Class`, `random_state=42`.
- `StandardScaler` yalnızca `Time` ve `Amount` sütunlarına, **yalnızca normal eğitim işlemlerine fit edilir**
  ve test kümesine uygulanır (veri sızıntısı önlenir).
- Eğitim kümesi yalnızca normal işlemlerden oluşur; test kümesi hem normal hem fraud içerir.

### 3.2 Modeller (`scripts/modeling.py`)

| Model | Fikir | Ayar |
|---|---|---|
| Isolation Forest | Ağaçlarla izolasyon; anormal noktalar daha az bölmede ayrılır | `n_estimators=100`, `contamination=0.001` |
| Local Outlier Factor | Yerel yoğunluğu komşularıyla karşılaştırır | `n_neighbors=20`, `novelty=True` |
| One-Class SVM | Normal verinin sınırını öğrenir | `kernel=rbf`, `nu=0.001` |

LOF ve One-Class SVM büyük veride yavaştır; eğitimde örneklem üst sınırı kullanılır
(LOF 50.000, OCSVM 20.000; `scripts/config.py`).

### 3.3 Değerlendirme (`scripts/evaluation.py`)
Fraud sınıfı için Precision, Recall, F1; ayrıca eşikten bağımsız PR-AUC ve ROC-AUC.
Confusion matrix'ler `predictions.csv`'den çizilir; modeller yeniden eğitilmez, böylece
grafikler ve `model_scores.json` birbiriyle tutarlıdır.

### 3.4 SQL (`scripts/sql_queries.py`)
Veri `transactions` tablosuna aktarılır; toplam işlem, fraud sayısı, sınıfa göre ortalama tutar ve
en yüksek 5 tutar sorgulanır.

### 3.5 Kullanıcı arayüzü (`scripts/cli_app.py`)
Etkileşimli menü ve `preprocess | train | evaluate | all | sql` alt komutları; `--demo` ile sentetik veri.

## 4. Doğrulama

`tests/test_pipeline.py` sentetik veriyle şunları doğrular: şema, eğitim kümesinin yalnızca normal
olması, test kümesinde fraud bulunması, ölçekleyicinin yalnızca eğitime fit edilmesi, metriklerin
geçerli aralıkta olması, tüm çıktı dosyalarının oluşması ve SQL sayımlarının CSV ile eşleşmesi.

## 5. İlk Sürümün Sonuçları ve Çıkarılan Dersler

İlk sürümde modeller **tüm veriyle (fraud dahil) eğitilip aynı veri üzerinde** değerlendirilmişti
(eğitim = test). Fraud sınıfı sonuçları:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Isolation Forest | 0,337 | 0,195 | 0,247 |
| Local Outlier Factor | 0,000 | 0,000 | 0,000 |
| One-Class SVM | 0,086 | 0,272 | 0,130 |

Gözlemler:
- Genel doğruluk %99,4–99,8 çıksa da fraud yakalama başarısı düşüktür; doğruluk bu veride anlamsızdır.
- LOF fraud'u hiç yakalayamamıştır. Eğitim verisinin kendisi üzerinde `novelty=True` ile tahmin
  yapmak, LOF için güvenilir bir değerlendirme değildir.
- Eğitim ve test aynı olduğu, `random_state` verilmediği ve grafikler için modeller yeniden
  eğitildiği için sonuçlar tekrarlanabilir değildi.

Bu bulgular nedeniyle yöntem 3. bölümdeki hâline getirilmiştir. Yeni düzenle üretilen sonuçlar
`results/summary.md` dosyasındadır; **ilk sürümle doğrudan karşılaştırılamaz** (farklı bölme ve
eğitim kümesi).

## 6. Etik Değerlendirme

- Fraud'u kaçırmak (False Negative) doğrudan mali zarar demektir; bu yüzden Recall önemlidir.
- Yanlış alarmlar (False Positive) masum müşterinin işleminin engellenmesine yol açar; Precision
  da ihmal edilmemelidir. İki hata arasındaki denge iş kuralıyla (maliyet) belirlenmelidir.
- Karar sistemleri şeffaf ve açıklanabilir olmalıdır; V1–V28 anonim olduğundan bu veride
  özellik bazlı açıklama sınırlıdır.
- Model çıktısı tek başına işlem engelleme kararı olmamalı, insan incelemesiyle desteklenmelidir.

## 7. İlgili Faz 1 Eğitimleri

| Kurs | Projedeki yeri |
|---|---|
| Anomaly Detection in Python | Modellerin seçimi ve yapılandırılması (doğrudan kullanıldı) |
| Applying SQL to Real-World Problems | SQLite aktarımı ve sorgular (doğrudan kullanıldı) |
| AI Ethics | Etik değerlendirme bölümü (doğrudan kullanıldı) |
| Big Data Fundamentals with PySpark | Büyük veriyi parça parça işleme yaklaşımı (arka plan) |
| ARIMA Models in Python, Advanced NLP with spaCy, Business Process Analytics in R | Bu projede doğrudan kullanılmadı |

## 8. Öğrenilenler

- Dengesiz veride doğru metrik seçimi ve eğitim/test ayrımı (sızıntı önleme).
- Gözetimsiz anomali tespit algoritmalarının farklılıkları ve ölçeklenebilirlik sınırları.
- Tekrarlanabilir, modüler ve test edilebilir proje yapısı kurma; CLI ve SQL entegrasyonu.

## 9. Olası Geliştirmeler

Hiperparametre araması, eşik/`contamination` seçimi, gözetimli yöntemlerle karşılaştırma
(dengesizlik düzeltmeli lojistik regresyon, gradient boosting), basit bir arayüz (GUI).
