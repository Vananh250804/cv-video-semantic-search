import json
import pathlib
import pandas as pd

root = pathlib.Path("data/bdd100k")
with open(root / "samples.json") as f:
    samples = json.load(f)["samples"]


def lab(s, key):
    v = s.get(key)
    return v.get("label") if isinstance(v, dict) else v


rows = [{"image": pathlib.Path(s["filepath"]).name,
         "weather": lab(s, "weather"),
         "scene": lab(s, "scene"),
         "timeofday": lab(s, "timeofday")} for s in samples]
pathlib.Path("artifacts").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("artifacts/all_labels.csv", index=False)
print(f"Đã lưu artifacts/all_labels.csv ({len(rows)} ảnh, đủ 3 nhãn gốc)")
