import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

res = pathlib.Path("results"); res.mkdir(exist_ok=True)
counts = pd.read_csv("results/class_counts.csv")
splits = pd.read_csv("artifacts/splits.csv")
USED = {
    "weather":   ["clear", "overcast", "snowy", "rainy"],
    "scene":     ["city street", "highway", "residential"],
    "timeofday": ["daytime", "night", "dawn/dusk"],
}

# Hình 1: phân bố nhãn gốc của 10.000 ảnh (xanh = dùng, xám = bỏ; đường đỏ = mốc 500)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, field in zip(axes, USED):
    d = counts[counts["field"] == field].sort_values("count", ascending=False)
    labels = d["label"].astype(str).tolist()
    colors = ["#2a7ab0" if l in USED[field] else "#b8b8b8" for l in labels]
    ax.bar(labels, d["count"], color=colors)
    ax.axhline(500, ls="--", color="red", lw=1)
    ax.set_yscale("log")
    ax.set_title(field)
    ax.tick_params(axis="x", rotation=40)
    for t in ax.get_xticklabels():
        t.set_ha("right")
axes[0].set_ylabel("Số ảnh (thang log)")
fig.suptitle("Phân bố nhãn gốc trong 10.000 ảnh (xanh: dùng, xám: bỏ, đỏ: mốc 500 ảnh)")
fig.tight_layout()
fig.savefig(res / "fig_original_distribution.png", dpi=150)

# Hình 2: số ảnh mỗi lớp trong train/val/test
pv = splits.pivot_table(index=["task", "label"], columns="split",
                        values="image", aggfunc="count")[["train", "val", "test"]]
ax = pv.plot(kind="bar", stacked=True, figsize=(11, 5))
ax.set_ylabel("Số ảnh")
ax.set_title("Số ảnh mỗi lớp theo tập train/val/test")
plt.tight_layout()
plt.savefig(res / "fig_split_counts.png", dpi=150)

# Hình 3: 5 ảnh mẫu cho mỗi lớp
rng = np.random.default_rng(0)
classes = [(t, l) for t in USED for l in USED[t]]
fig, axes = plt.subplots(len(classes), 5, figsize=(15, 2.3 * len(classes)))
for r, (t, l) in enumerate(classes):
    pool = splits[(splits.task == t) & (splits.label == l) &
                  (splits.split == "train")]["image"].to_numpy()
    for c, name in enumerate(rng.choice(pool, 5, replace=False)):
        ax = axes[r, c]
        ax.imshow(Image.open(f"data/images_640/{name}"))
        ax.axis("off")
        if c == 0:
            ax.set_title(f"{t}: {l}", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(res / "fig_samples.png", dpi=110)
print("Đã lưu 3 hình trong thư mục results/")
