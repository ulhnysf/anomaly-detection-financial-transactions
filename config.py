"""Proje genelinde kullanılan yollar ve sabit parametreler."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Deney parametreleri ---------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.30
SCALE_COLUMNS = ["Time", "Amount"]  # V1-V28 zaten PCA çıktısıdır
CONTAMINATION = 0.001               # modellerin beklediği anomali oranı

# O(n^2) karmaşıklık nedeniyle yavaş modeller için eğitim örneklem üst sınırı
MAX_TRAIN_SAMPLES = {
    "Isolation Forest": None,       # None = tüm normal eğitim verisi
    "Local Outlier Factor": 50_000,
    "One-Class SVM": 20_000,
}


@dataclass(frozen=True)
class Paths:
    """Girdi/çıktı dosya yolları. Tek yerden yönetildiği için script'ler
    hangi klasörden çalıştırılırsa çalıştırılsın doğru dosyaları bulur."""

    raw_csv: Path
    processed_dir: Path
    results_dir: Path
    db_path: Path

    @property
    def train_csv(self) -> Path:
        return self.processed_dir / "train_normal.csv"

    @property
    def test_csv(self) -> Path:
        return self.processed_dir / "test.csv"

    @property
    def scores_json(self) -> Path:
        return self.results_dir / "model_scores.json"

    @property
    def predictions_csv(self) -> Path:
        return self.results_dir / "predictions.csv"

    @classmethod
    def default(cls) -> "Paths":
        return cls(
            raw_csv=ROOT / "data" / "creditcard.csv",
            processed_dir=ROOT / "data" / "processed",
            results_dir=ROOT / "results",
            db_path=ROOT / "data" / "transactions.db",
        )

    @classmethod
    def demo(cls) -> "Paths":
        return cls(
            raw_csv=ROOT / "data" / "demo" / "demo_creditcard.csv",
            processed_dir=ROOT / "data" / "demo" / "processed",
            results_dir=ROOT / "results" / "demo",
            db_path=ROOT / "data" / "demo" / "transactions.db",
        )

    @classmethod
    def under(cls, base: Path) -> "Paths":
        """Testler için: her şeyi verilen klasörün altına yazar."""
        base = Path(base)
        return cls(
            raw_csv=base / "creditcard.csv",
            processed_dir=base / "processed",
            results_dir=base / "results",
            db_path=base / "transactions.db",
        )
