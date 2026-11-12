import json, collections, pathlib
import pandas as pd

root = pathlib.Path("data/bdd100k")
with open(root / "samples.json") as f:
    samples = json.load(f)["samples"]

print("So mau:", len(samples))
print("Cac truong cua mau dau tien:", list(samples[0].keys()))
print("filepath mau dau:", samples[0]["filepath"],
      "| ton tai:", (root / samples[0]["filepath"]).exists())

rows = []
for field in ["weather", "timeofday", "scene"]:
    cnt = collections.Counter()
    for s in samples:
        v = s.get(field)
        label = v.get("label") if isinstance(v, dict) else v
        cnt[label] += 1
    print(f"\n=== {field} ===")
    for label, n in cnt.most_common():
        flag = "OK (>=500)" if n >= 500 else "THIEU (<500)"
        print(f"  {str(label):20s} {n:6d}  {flag}")
        rows.append({"field": field, "label": label, "count": n})

pathlib.Path("results").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("results/class_counts.csv", index=False)
print("\nDa luu results/class_counts.csv")
