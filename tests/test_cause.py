"""Cause model: features from FOD columns, honest last-year score, probabilities per cause."""
import numpy as np
import pandas as pd

import jamra_cause.model as cm


def synth(n=4000, seed=3):
    """Lightning in hot dry summer storms on trees; debris burning in wet cool months on crops."""
    rng = np.random.default_rng(seed)
    natural = rng.random(n) < 0.5
    return pd.DataFrame({
        "label": np.where(natural, "طبيعي (برق)", "حرق مخلفات وزراعي"),
        "year": rng.choice([2018, 2019, 2020], n),
        "month": np.where(natural, rng.choice([6, 7, 8], n), rng.choice([2, 3, 4, 10, 11], n)),
        "weekend": rng.integers(0, 2, n),
        "tmax": np.where(natural, 33, 15) + rng.normal(0, 3, n),
        "rhmin": np.where(natural, 15, 45) + rng.normal(0, 5, n),
        "wind": rng.uniform(5, 30, n), "rain": rng.gamma(1, 1, n), "rain5": rng.gamma(1, 1, n),
        "cover": np.where(natural, "tree", "crop")})


def test_learns_and_scores_last_year_only():
    m = cm.CauseModel(synth())
    assert m.eval["test_year"] == 2020 and m.eval["train_years"] == "2018–2019"
    assert m.eval["accuracy"] > 0.9 > m.eval["chance"]
    p = m.predict(month=7, weekend=False, tmax=35, rhmin=12, wind=20, rain=0, rain5=0, cover="tree")
    assert p[0]["cause"] == "طبيعي (برق)"
    assert [r["p"] for r in p] == sorted((r["p"] for r in p), reverse=True)
    assert abs(sum(r["p"] for r in p) - 1) < 0.01
    assert m.predict(3, True, 14, 50, 10, 1, 1, cover=None)[0]["cause"] == "حرق مخلفات وزراعي"  # unknown cover ok


def test_build_features_from_fod_columns(tmp_path):
    csv = tmp_path / "2020.csv"
    pd.DataFrame({"NWCG_GENERAL_CAUSE": ["Natural", "Smoking", "Missing data/not specified/undetermined"],
                  "DISCOVERY_DATE": ["2020-07-04", "2020-03-02", "2020-01-01"],
                  "tmmx": [310.15, 290.15, 280.0], "rmin": [12, 40, 50], "vs": [5, 2, 1],
                  "pr": [0, 1.5, 0], "pr_5D_mean": [0, 2, 0], "Land_Cover": ["42.0", "81", "11"]}).to_csv(csv, index=False)
    t = cm.build_features([str(csv)], str(tmp_path / "f.parquet"))
    assert list(t["label"]) == ["طبيعي (برق)", "إهمال وترفيه"]           # unknown cause dropped
    assert abs(t["tmax"].iloc[0] - 37.0) < 1e-9 and abs(t["wind"].iloc[0] - 18.0) < 1e-9   # K -> °C, m/s -> km/h
    assert list(t["cover"]) == ["tree", "crop"] and list(t["weekend"]) == [1, 0]           # 2020-07-04 is a Saturday


def test_jordan_weekend_and_cover():
    assert cm.jordan_weekend("2026-11-13") and cm.jordan_weekend("2026-11-14")   # Friday, Saturday
    assert not cm.jordan_weekend("2026-11-15")
    from h3.api import basic_int as h3
    lc = pd.DataFrame({"h3": [np.uint64(h3.latlng_to_cell(32.3, 35.8, 5))], "lc_tree": [0.6], "lc_crop": [0.3]})
    assert cm.dominant_cover(32.3, 35.8, lc) == "tree"
    assert cm.dominant_cover(0.0, 0.0, lc) is None and cm.dominant_cover(32.3, 35.8, None) is None
