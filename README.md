# jamra-cause — likely fire cause

Team TFC · NASA Space Apps 2026 · part of [JAMRA](https://jamra.tfs.ink).

[العربية ↓](#العربية)

## What it does
Given a place and a day, it returns a probability for each of five grouped causes:
**natural (lightning)**, **debris and agricultural burning**, **arson**, **equipment, vehicles and power lines**,
**negligence and recreation** (smoking, campfires, fireworks, minors, firearms).

It learns from the **US Fire Program Analysis Fire-Occurrence Database (FPA-FOD)**, where every fire has an
official cause, through the **FPA FOD-Attributes** release that adds each fire's weather and land cover.
It uses **only features we can also measure in Jordan**, so the model can be applied there:
month, weekend (US Sat–Sun; Jordan Fri–Sat via `jordan_weekend`), max temperature, min humidity, wind, rain of the day
and 5-day mean rain, and land cover mapped to the same classes as ESA WorldCover (FireCal's `landcover.parquet`).
Fires with unknown cause are left out of training.

Model: scikit-learn `HistGradientBoostingClassifier` with balanced class weights.
Honest score: trained on every year but the last, scored on the last year it never saw; then retrained on all years.

## Results
Built from the FPA FOD-Attributes files for **2018–2020: 134,177 US fires with a known cause**
(debris/agricultural 35 %, equipment/vehicles/power 22 %, natural 17 %, negligence/recreation 14 %, arson 12 %).
Trained on 2018–2019, scored on the 44,776 fires of 2020 it never saw:

| | held-out 2020 |
|---|---|
| accuracy | **42.0 %** |
| macro-F1 (5 classes) | **0.407** |
| always guessing the largest class | 31.9 % |
| true cause among the model's first two | **66.4 %** |
| always guessing the two largest classes | 56.6 % |

Better than guessing, far from certain: read the output as a ranked list of likely causes, not an answer.
More years (`download_fpa.py` without `--years`, 1992–2020, ~2 million fires) can be added; the numbers above are for 2018–2020 only.

## Install
```bash
uv sync
uv run pytest -q
```

## Minimal example
```python
import pandas as pd
import jamra_cause as jc

table = pd.read_parquet("data/cause_features.parquet")   # jamra-cause build
m = jc.CauseModel(table)
print(m.eval)                                            # accuracy, macro-F1, chance on the held-out year
print(m.predict(month=7, weekend=jc.jordan_weekend("2024-06-17"), tmax=36, rhmin=14, wind=22,
                rain=0, rain5=0, cover="crop"))          # [{'cause': ..., 'p': ...}, ...] highest first
```

## Data
No data is committed.
```bash
python scripts/download_fpa.py --years 2018 2019 2020   # or no --years for 1992–2020 (~5 GB)
uv run jamra-cause build --csv "data_fpa/*.csv"          # -> data/cause_features.parquet + score
```
Sources: **FPA FOD-Attributes** (Zenodo [10.5281/zenodo.8381129](https://doi.org/10.5281/zenodo.8381129), CC BY 4.0),
built on **FPA-FOD** (USDA Forest Service, Short 2022). Land-cover classes match **ESA WorldCover** as used by FireCal.

## Honest limits
- **Learned on US fires.** Burning habits, farming and land use in Jordan differ; the transfer to Jordan is **not measured**.
  JAMRA shows it as extra evidence next to its rule-based investigation, never as a verdict.
- The score is for US fires of one held-out year, with a coarse set of features (no distance to roads or towns,
  no human-activity layers), so causes that depend on people are hard to separate.
- Five grouped classes hide finer causes; "unknown cause" fires (a large share of FPA-FOD) are not used.

---

<div dir="rtl">

## العربية

# jamra-cause — سبب الحريق المرجّح

### ماذا يفعل
لمكان ويوم معيّنين يعطي احتمالاً لكل سبب من خمسة: **طبيعي (برق)**، **حرق مخلفات وزراعي**، **متعمّد**،
**معدات ومركبات وخطوط**، **إهمال وترفيه**.
يتعلّم من قاعدة حرائق الولايات المتحدة **FPA-FOD** (كل حريق مع سببه الرسمي) عبر إصدار **FPA FOD-Attributes**
الذي يضيف طقس كل حريق وأرضه. ويستعمل **فقط ما نستطيع قياسه في الأردن أيضاً**: الشهر، نهاية الأسبوع، الحرارة،
الرطوبة، الريح، المطر، ونوع الأرض بنفس فئات ESA WorldCover.
القياس صادق: التدريب على كل السنين إلا الأخيرة، والقياس على السنة الأخيرة التي لم يرها.

### النتائج
من ملفات 2018–2020: **134,177 حريقاً أمريكياً معروف السبب**. تدريب على 2018–2019 وقياس على 44,776 حريقاً من 2020 لم يرها:
الدقة **42.0%**، وmacro-F1 **0.407**، مقابل 31.9% لو خمّنا دائماً السبب الأكثر شيوعاً.
والسبب الصح ضمن أول اتنين **66.4%** من المرات، مقابل 56.6% لو خمّنا دائماً أكثر سببين.
أفضل من التخمين لكنه بعيد عن اليقين: اقرأ النتيجة كترتيب للأسباب المحتملة لا كجواب.
يمكن إضافة سنوات أكثر (1992–2020، نحو مليوني حريق)؛ الأرقام أعلاه لـ 2018–2020 فقط.

### التثبيت
```bash
uv sync
uv run pytest -q
```

### البيانات
لا نرفع بيانات. `scripts/download_fpa.py` ينزّل الملفات من Zenodo، ثم `jamra-cause build` يبني جدول الميزات.
المصادر: FPA FOD-Attributes (CC BY 4.0) فوق FPA-FOD (USDA Forest Service).

### حدود بصراحة
- **متعلَّم على حرائق أمريكا.** عادات الحرق والزراعة في الأردن مختلفة، والنقل إلى الأردن **غير مقاس**؛ لذلك تعرضه جمرة
  كدليل إضافي بجانب قواعد التحقيق، لا كحكم.
- القياس على حرائق أمريكية لسنة واحدة، وبميزات محدودة (بلا بُعد عن الطرق أو المدن)، فالأسباب البشرية يصعب فصلها.
- الحرائق مجهولة السبب لا تدخل التدريب.

</div>

## License
MIT, see [LICENSE](LICENSE). Data keep their own terms: NASA FIRMS (open), Open-Meteo (CC BY 4.0), FPA FOD-Attributes (CC BY 4.0).
