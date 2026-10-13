# Anomaly Detection in Financial Transactions

Kredi kartı işlemlerinde dolandırıcılığı (fraud) **gözetimsiz anomali tespiti** ile bulmaya
çalışan, uçtan uca bir Python projesi: veri ön işleme → üç modelin karşılaştırılması →
değerlendirme/görselleştirme → SQL analizi → komut satırı arayüzü.

> İş Yeri Uygulaması (IYU 228) Faz 2 projesi · Hazırlayan: Uluhan Yusuf Dokay

## Özellikler

- **Üç model karşılaştırması:** Isolation Forest, Local Outlier Factor (LOF), One-Class SVM
- **Doğru değerlendirme düzeneği:** stratified eğitim/test bölmesi, yalnızca normal işlemlerle
  eğitim, ölçekleyiciyi sadece eğitim verisine fit etme (veri sızıntısı yok), sabit `random_state`
- **Dengesiz veriye uygun metrikler:** Precision / Recall / F1 (fraud sınıfı) + PR-AUC ve ROC-AUC
- **CLI:** etkileşimli menü ve alt komutlar (`python main.py all`)
- **SQL entegrasyonu:** veriyi SQLite'a aktarıp analiz sorguları çalıştırır
- **Otomatik testler:** `pytest` ile uçtan uca doğrulama (sentetik veriyle birkaç saniye)
- **Demo modu:** Kaggle verisi olmadan da çalıştırılabilir

## Kurulum

```bash
git clone https://github.com/<kullanici-adi>/<repo-adi>.git
cd <repo-adi>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Veri seti: `creditcard.csv` dosyasını Kaggle'dan indirip `data/` klasörüne koyun
(ayrıntı: [data/README.md](data/README.md)).

## Kullanım

```bash
python main.py                  # etkileşimli menü
python main.py all              # ön işleme + eğitim + değerlendirme
python main.py preprocess       # yalnızca ön işleme
python main.py train            # yalnızca model eğitimi
python main.py evaluate         # yalnızca grafikler/özet
python main.py sql              # SQLite aktarımı + SQL sorguları

python main.py all --demo       # sentetik demo verisiyle (veri indirmeden) dene
python -m pytest                # testleri çalıştır
```

## Proje Yapısı

```
.
├── main.py                 # giriş noktası
├── scripts/
│   ├── config.py           # yollar ve parametreler
│   ├── preprocessing.py    # kontrol, bölme, ölçekleme, sınıf dağılımı grafiği
│   ├── modeling.py         # model eğitimi ve test kümesi değerlendirmesi
│   ├── evaluation.py       # karşılaştırma grafiği, confusion matrix, özet tablo
│   ├── sql_queries.py      # CSV -> SQLite + SQL sorguları
│   ├── cli_app.py          # menü ve alt komutlar
│   └── make_demo_data.py   # sentetik demo verisi
├── tests/test_pipeline.py  # doğrulama testleri
├── docs/report.md          # teknik rapor
├── data/                   # veri (büyük dosyalar git dışı)
├── results/                # grafikler, model_scores.json, predictions.csv, summary.md
└── requirements.txt
```

## Yöntem (kısaca)

1. **Ön işleme:** eksik değer kontrolü, sınıf dağılımı grafiği, %70/%30 stratified bölme.
   Sadece `Time` ve `Amount` ölçeklenir (V1–V28 zaten PCA çıktısıdır).
2. **Modelleme:** modeller yalnızca **normal** eğitim işlemleriyle eğitilir; test kümesinde
   hem normal hem fraud işlemler bulunur. LOF ve One-Class SVM, karmaşıklıkları nedeniyle
   eğitimde örneklem üst sınırıyla çalışır (`scripts/config.py`).
3. **Değerlendirme:** fraud sınıfı için Precision/Recall/F1, eşikten bağımsız PR-AUC ve ROC-AUC;
   confusion matrix'ler kaydedilmiş tahminlerden çizilir (modeller yeniden eğitilmez).

Ayrıntılar: [docs/report.md](docs/report.md)

## Sonuçlar

Projeyi gerçek veriyle çalıştırdıktan sonra sonuçlar `results/` klasöründe oluşur
(`summary.md`, `model_comparison.png`, `confusion_matrix_*.png`). Bu çıktıları commit'leyerek
README'ye ekleyebilirsiniz:

```markdown
![Model karşılaştırması](results/model_comparison.png)
```

## Sınırlılıklar

- Veri seti çok dengesiz (%0,17 fraud); gözetimsiz modellerin Precision'ı düşük kalabilir.
- Hiperparametre araması yapılmamıştır (`contamination = 0.001` sabittir).
- V1–V28 anonim PCA bileşenleri olduğundan sonuçların iş yorumu sınırlıdır.
- Gözetimli yöntemler (ör. dengesizlik düzeltmeli lojistik regresyon / gradient boosting) ile
  karşılaştırma ileride yapılabilir.

## Lisans / Not

Bu proje IYU 228 dersi kapsamında bireysel eğitim ve değerlendirme amacıyla hazırlanmıştır.
Veri seti [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
tarafından yayımlanmıştır; kendi lisans koşullarına tabidir.
