"""Read the fr/cr xlsx, dedupe, and write a clean UTF-8 CSV + train/val split."""
import csv
import random

import openpyxl

SRC_XLSX = r"C:\Users\nolan\Downloads\fr_creole_guyanais_5000.xlsx"
OUT_ALL = "data_pairs.csv"
OUT_TRAIN = "data_train.csv"
OUT_VAL = "data_val.csv"

random.seed(42)


def load_pairs():
    wb = openpyxl.load_workbook(SRC_XLSX, read_only=True)
    ws = wb.worksheets[0]
    seen = set()
    pairs = []
    for fr, cr in ws.iter_rows(min_row=2, values_only=True):
        if not fr or not cr:
            continue
        fr, cr = fr.strip(), cr.strip()
        if (fr, cr) in seen:
            continue
        seen.add((fr, cr))
        pairs.append((fr, cr))
    return pairs


def main():
    pairs = load_pairs()
    print(f"Unique pairs: {len(pairs)}")

    with open(OUT_ALL, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fr", "cr"])
        w.writerows(pairs)

    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * 0.1))
    val, train = pairs[:n_val], pairs[n_val:]

    for path, rows in [(OUT_TRAIN, train), (OUT_VAL, val)]:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fr", "cr"])
            w.writerows(rows)

    print(f"Train: {len(train)}  Val: {len(val)}")


if __name__ == "__main__":
    main()
