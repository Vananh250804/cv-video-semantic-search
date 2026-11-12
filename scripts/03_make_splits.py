import json, pathlib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

SEED = 42
N_PER_CLASS = 600          # so anh moi lop (>= 500 theo yeu cau)
TRAIN, VAL = 0.70, 0.15    # phan con lai (15%) la test
TARGET_W = 640             # chieu rong anh sau khi thu nho
TASKS = {
    "weather":   ["clear", "overcast", "snowy", "rainy"],
    "scene":     ["city street", "highway", "residential"],
    "timeofday": ["daytime", "night", "dawn/dusk"],
}
ROOT = pathlib.Path("data/bdd100k")
OUT_IMG = pathlib.Path("data/images_640")

rng = np.random.default_rng(SEED)

with open(ROOT / "samples.json") as f:
    samples = json.load(f)["samples"]

recs = []
for s in samples:
    rec = {"filepath": s["filepath"]}
    for t in TASKS:
        v = s.get(t)
        rec[t] = v.get("label") if isinstance(v, dict) else v
    recs.append(rec)
df = pd.DataFrame(recs)

rows = []
for task, classes in TASKS.items():
    for label in classes:
        pool = df.loc[df[task] == label, "filepath"].to_numpy()
        n = min(N_PER_CLASS, len(pool))
        chosen = rng.choice(pool, size=n, replace=False)   # da xao tron
        n_tr, n_va = int(n * TRAIN), int(n * VAL)
        for i, fp in enumerate(chosen):
            split = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
            rows.append({"image": pathlib.Path(fp).name, "filepath": fp,
                         "task": task, "label": label, "split": split})

splits = pd.DataFrame(rows)
splits.to_csv("artifacts/splits.csv", index=False)

print("=== So anh moi lop theo tap ===")
print(splits.pivot_table(index=["task", "label"], columns="split",
                         values="image", aggfunc="count")[["train", "val", "test"]])
uniq = sorted(splits["image"].unique())
print(f"\nTong luot gan nhan: {len(splits)} | so anh khac nhau can embed: {len(uniq)}")

# Thu nho anh (bo qua anh da lam)
OUT_IMG.mkdir(parents=True, exist_ok=True)
src = dict(zip(splits["image"], splits["filepath"]))
for name in tqdm(uniq, desc="Thu nho anh"):
    dst = OUT_IMG / name
    if dst.exists():
        continue
    im = Image.open(ROOT / src[name]).convert("RGB")
    w, h = im.size
    im = im.resize((TARGET_W, round(h * TARGET_W / w)), Image.LANCZOS)
    im.save(dst, quality=90)
print("Xong. Anh nam o", OUT_IMG)
