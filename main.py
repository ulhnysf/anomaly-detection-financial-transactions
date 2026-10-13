"""Giriş noktası.

Kullanım:
    python main.py                 # etkileşimli menü
    python main.py all             # tüm adımlar (data/creditcard.csv gerekir)
    python main.py all --demo      # sentetik demo verisiyle uçtan uca çalıştır
"""

from scripts.cli_app import main

if __name__ == "__main__":
    main()
