"""CSV verisini SQLite veritabanına aktarır ve örnek SQL sorguları çalıştırır."""

import sqlite3
from contextlib import closing

import pandas as pd

from .config import Paths

QUERIES = {
    "Toplam işlem sayısı": "SELECT COUNT(*) FROM transactions;",
    "Fraud (Class = 1) işlem sayısı": "SELECT COUNT(*) FROM transactions WHERE Class = 1;",
    "Sınıfa göre ortalama tutar": (
        "SELECT Class, ROUND(AVG(Amount), 2) FROM transactions GROUP BY Class ORDER BY Class;"
    ),
    "En yüksek tutarlı 5 işlem": (
        "SELECT Amount, Class FROM transactions ORDER BY Amount DESC LIMIT 5;"
    ),
}


def csv_to_sqlite(csv_path, db_path, chunksize: int = 50_000) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        # Büyük CSV'yi belleğe tek seferde yüklememek için parça parça yaz
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
            chunk.to_sql("transactions", conn, if_exists="replace" if i == 0 else "append", index=False)
        conn.commit()
    print(f"Veri SQLite veritabanına aktarıldı: {db_path}")


def run_queries(db_path) -> dict:
    out = {}
    with closing(sqlite3.connect(db_path)) as conn:
        for title, sql in QUERIES.items():
            rows = conn.execute(sql).fetchall()
            out[title] = rows
            print(f"\n{title}:")
            for row in rows:
                print("  ", row if len(row) > 1 else row[0])
    return out


def main(paths: Paths | None = None):
    paths = paths or Paths.default()
    if not paths.raw_csv.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {paths.raw_csv}")
    csv_to_sqlite(paths.raw_csv, paths.db_path)
    return run_queries(paths.db_path)


if __name__ == "__main__":
    main()
