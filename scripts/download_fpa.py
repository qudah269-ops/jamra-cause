"""Download the FPA FOD-Attributes yearly CSVs (Zenodo 10.5281/zenodo.8381129, CC BY 4.0) into data_fpa/.

    python scripts/download_fpa.py                 # 1992-2020, ~5 GB
    python scripts/download_fpa.py --years 2018 2019 2020
Existing files are skipped; each lands under its year name (2020.csv) as `jamra-cause build` expects.
"""
import argparse
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/api/records/8381129/files/{y}_FPA_FOD_cons.csv/content"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data_fpa")
    p.add_argument("--years", nargs="+", type=int, default=list(range(1992, 2021)))
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for y in a.years:
        dest = Path(a.out) / f"{y}.csv"
        if dest.exists():
            continue
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(URL.format(y=y), timeout=300) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.rename(dest)  # never leave a half file under the real name
        print("ok", dest)


if __name__ == "__main__":
    main()
