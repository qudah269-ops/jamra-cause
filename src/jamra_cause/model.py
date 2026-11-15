"""
jamra_cause.model (JAMRA cause.py) — سبب الحريق المرجّح، متعلَّم من 2 مليون حريق أمريكي معروف سببها

المصدر: FPA FOD-Attributes (Zenodo 10.5281/zenodo.8381129)، فوق قاعدة حرائق الولايات المتحدة
FPA-FOD (USDA Forest Service): كل حريق مع سببه الرسمي، وطقسه، وأرضه.

نتعلّم بس من اللي بنقدر نقيسه بالأردن كمان، عشان الموديل ينفع هون:
  الشهر · نهاية الأسبوع (بأمريكا سبت-أحد، بالأردن جمعة-سبت) · حرارة اليوم العظمى · أدنى رطوبة ·
  الريح · مطر اليوم ومتوسط مطر 5 أيام · نوع الأرض (نفس فئات ESA WorldCover).
والأسباب مجمّعة بخمس فئات واضحة. الحرائق مجهولة السبب ما بتدخل التدريب.

⚠ بصراحة: الموديل متعلَّم على حرائق أمريكا. عادات الحرق والزراعة بالأردن غير، فبنعرضه كدليل
  إضافي جنب قواعد التحقيق، مش كحكم. القياس تحت (سنة 2020 ما شافها التدريب) على حرائق أمريكا.

  jamra-cause build   → data/cause_features.parquet (مرة وحدة، من ملفات data_fpa/*.csv)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

CLASSES = {
    "Natural": "طبيعي (برق)",
    "Debris and open burning": "حرق مخلفات وزراعي",
    "Arson/incendiarism": "متعمّد",
    "Equipment and vehicle use": "معدات ومركبات وخطوط",
    "Power generation/transmission/distribution": "معدات ومركبات وخطوط",
    "Railroad operations and maintenance": "معدات ومركبات وخطوط",
    "Recreation and ceremony": "إهمال وترفيه",
    "Smoking": "إهمال وترفيه",
    "Misuse of fire by a minor": "إهمال وترفيه",
    "Fireworks": "إهمال وترفيه",
    "Firearms and explosives use": "إهمال وترفيه",
}
# NLCD land cover (FPA FOD) → the same classes as ESA WorldCover in FireCal's landcover.parquet
NLCD = {11: "water", 21: "built", 22: "built", 23: "built", 24: "built", 31: "bare", 41: "tree", 42: "tree", 43: "tree",
        52: "shrub", 71: "grass", 81: "crop", 82: "crop", 90: "wetland", 95: "wetland"}
COVERS = ["tree", "shrub", "grass", "crop", "built", "bare", "water", "wetland"]
FEATS = ["msin", "mcos", "weekend", "tmax", "rhmin", "wind", "rain", "rain5"] + [f"lc_{c}" for c in COVERS]


def build_features(csv_paths, out_path):
    """The FPA FOD-Attributes yearly CSVs → one small table of transferable features + the grouped cause."""
    import duckdb
    files = ", ".join(f"'{p}'" for p in csv_paths)
    con = duckdb.connect()          # keep a name: an unnamed connection closes before .df()
    df = con.sql(f"""
        select NWCG_GENERAL_CAUSE as cause, try_cast(DISCOVERY_DATE as date) as date,
               try_cast(tmmx as double) - 273.15 as tmax, try_cast(rmin as double) as rhmin,
               try_cast(vs as double) * 3.6 as wind, try_cast(pr as double) as rain, try_cast(pr_5D_mean as double) as rain5,
               try_cast(try_cast(Land_Cover as double) as int) as nlcd
        from read_csv([{files}], all_varchar = true, sample_size = -1)""").df()
    df = df[df["cause"].isin(CLASSES)].dropna(subset=["date"])
    df["label"] = df["cause"].map(CLASSES)
    d = pd.to_datetime(df["date"])
    df["year"], df["month"], df["dow"] = d.dt.year, d.dt.month, d.dt.day_of_week
    df["weekend"] = df["dow"].isin([5, 6]).astype(int)                      # US weekend: Saturday, Sunday
    df["cover"] = df["nlcd"].map(NLCD)
    df = df.drop(columns=["cause", "date", "dow", "nlcd"])
    df.to_parquet(out_path, index=False)
    return df


def _prep(df):
    m = df["month"].astype(float)
    x = pd.DataFrame({"msin": np.sin(2 * np.pi * m / 12), "mcos": np.cos(2 * np.pi * m / 12),
                      "weekend": df["weekend"].astype(float), "tmax": df["tmax"], "rhmin": df["rhmin"],
                      "wind": df["wind"], "rain": df["rain"], "rain5": df["rain5"]}, index=df.index)
    for c in COVERS:
        x[f"lc_{c}"] = (df["cover"] == c).astype(float).where(df["cover"].notna())
    return x[FEATS]


class CauseModel:
    """Trains on every year but the last; scores the last year honestly; then trains on all of it."""

    def __init__(self, table):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import accuracy_score, f1_score
        mk = lambda: HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=63,
                                                    class_weight="balanced", random_state=7)
        last = int(table["year"].max())
        tr, te = table[table["year"] < last], table[table["year"] == last]
        m = mk().fit(_prep(tr), tr["label"])
        p = m.predict(_prep(te))
        share = te["label"].value_counts(normalize=True)
        self.eval = {"train_years": f"{int(tr['year'].min())}–{last - 1}", "test_year": last, "fires_test": int(len(te)),
                     "accuracy": round(float(accuracy_score(te["label"], p)), 3),
                     "macro_f1": round(float(f1_score(te["label"], p, average="macro")), 3),
                     "chance": round(float(share.max()), 3), "classes": sorted(table["label"].unique()),
                     "fires_train": int(len(table)), "note": "measured on US fires; transfer to Jordan not measured"}
        self.model = mk().fit(_prep(table), table["label"])
        self.classes = list(self.model.classes_)

    def predict(self, month, weekend, tmax, rhmin, wind, rain, rain5, cover=None):
        """One place/day → probability per cause, highest first. Units: °C, %, km/h, mm."""
        row = pd.DataFrame([{"month": month, "weekend": int(weekend), "tmax": tmax, "rhmin": rhmin, "wind": wind,
                             "rain": rain, "rain5": rain5, "cover": cover}])
        p = self.model.predict_proba(_prep(row))[0]
        return sorted(({"cause": c, "p": round(float(v), 3)} for c, v in zip(self.classes, p)), key=lambda r: -r["p"])


def jordan_weekend(day):
    """Jordan's weekend is Friday and Saturday (the model learned the US one, Saturday and Sunday)."""
    return pd.Timestamp(day).dayofweek in (4, 5)


def dominant_cover(lat, lon, landcover):
    """Main land type of the ~252 km² cell at a point, from FireCal's ESA WorldCover table (None if unknown)."""
    from h3.api import basic_int as h3
    if landcover is None:
        return None
    row = landcover[landcover["h3"] == np.uint64(h3.latlng_to_cell(lat, lon, 5))]
    if row.empty:
        return None
    cols = [f"lc_{c}" for c in COVERS if f"lc_{c}" in row]
    return row[cols].iloc[0].astype(float).idxmax()[3:]


def main(argv=None):
    """build: FPA FOD-Attributes CSVs -> data/cause_features.parquet, then train and print the honest score.
    eval:  train from an existing features parquet and print the score."""
    import argparse
    import glob
    import json
    p = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--csv", default="data_fpa/*.csv", help="glob of the yearly FPA FOD-Attributes CSVs")
    b.add_argument("--out", default="data/cause_features.parquet")
    e = sub.add_parser("eval")
    e.add_argument("--features", default="data/cause_features.parquet")
    a = p.parse_args(argv)
    if a.cmd == "build":
        src = sorted(glob.glob(a.csv))
        if not src:
            raise SystemExit(f"no CSV matches {a.csv}: run scripts/download_fpa.py first")
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        t = build_features(src, a.out)
        print(len(t), "fires ->", a.out)
    else:
        t = pd.read_parquet(a.features)
    print(json.dumps(CauseModel(t).eval, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
