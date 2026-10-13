"""Komut satırı arayüzü: etkileşimli menü ve alt komutlar."""

import argparse

from . import evaluation, make_demo_data, modeling, preprocessing, sql_queries
from .config import Paths


def run_all(paths: Paths) -> None:
    preprocessing.main(paths)
    modeling.main(paths)
    evaluation.main(paths)
    print("\nTüm adımlar başarıyla tamamlandı!")


ACTIONS = {
    "preprocess": ("Veriyi ön işle (böl + ölçekle)", preprocessing.main),
    "train": ("Modelleri eğit ve test kümesinde değerlendir", modeling.main),
    "evaluate": ("Skorları ve grafikleri üret", evaluation.main),
    "all": ("Tüm adımları sırayla çalıştır", run_all),
    "sql": ("Veriyi SQLite'a aktar ve SQL sorguları çalıştır", sql_queries.main),
}
MENU_ORDER = ["preprocess", "train", "evaluate", "all", "sql"]


def _safe_run(func, paths: Paths) -> None:
    """Menüde bir adım hata verirse (ör. veri dosyası yok) program kapanmasın."""
    try:
        func(paths)
    except (FileNotFoundError, ValueError) as err:
        print(f"\n[HATA] {err}")


def menu(paths: Paths) -> None:
    while True:
        print("\n=== ANOMALİ TESPİT SİSTEMİ ===")
        for i, key in enumerate(MENU_ORDER, start=1):
            print(f"{i}. {ACTIONS[key][0]}")
        print("0. Çıkış")
        try:
            choice = input("Seçiminiz: ").strip()
        except EOFError:
            break
        if choice == "0":
            print("Programdan çıkılıyor.")
            break
        if choice.isdigit() and 1 <= int(choice) <= len(MENU_ORDER):
            _safe_run(ACTIONS[MENU_ORDER[int(choice) - 1]][1], paths)
        else:
            print("Geçersiz seçim! Tekrar deneyin.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Finansal işlemlerde anomali tespiti")
    p.add_argument("command", nargs="?", choices=list(ACTIONS),
                   help="Çalıştırılacak adım (verilmezse etkileşimli menü açılır)")
    p.add_argument("--demo", action="store_true",
                   help="Kaggle verisi yerine sentetik demo verisini kullan")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    paths = Paths.demo() if args.demo else Paths.default()
    if args.demo and not paths.raw_csv.exists():
        make_demo_data.main(paths)
    if args.command:
        ACTIONS[args.command][1](paths)
    else:
        menu(paths)
