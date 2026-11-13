import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

E = pathlib.Path("artifacts/embeddings")
z = np.load(E / "image_embeddings_qwen2b_768.npz", allow_pickle=False)
idx = {n: i for i, n in enumerate(z["names"].tolist())}
IMG = z["vectors"]
t = np.load(E / "text_prompts_qwen2b_768.npz", allow_pickle=False)
splits = pd.read_csv("artifacts/splits.csv")
splits = splits[splits["image"].isin(idx)]
pathlib.Path("results").mkdir(exist_ok=True)


def average_precision(rel):
    """AP: trung bình độ chính xác tại mỗi vị trí có kết quả đúng (rel là mảng 0/1 theo thứ tự xếp hạng)."""
    hits = np.cumsum(rel)
    prec = hits / (np.arange(len(rel)) + 1)
    return float((prec * rel).sum() / max(rel.sum(), 1))


EXAMPLES = [("weather", "rainy"), ("timeofday", "night"), ("scene", "highway")]
rows, ex_data = [], {}
for task, g in splits.groupby("task", sort=False):
    g = g.reset_index(drop=True)
    X = IMG[[idx[n] for n in g["image"]]]
    classes = sorted(g["label"].unique())
    chance = 1 / len(classes)
    for c in classes:
        m = (t["tasks"] == task) & (t["labels"] == c)
        q = t["vectors"][m].mean(axis=0)
        q /= np.linalg.norm(q)
        order = np.argsort(-(X @ q))                 # xếp hạng ảnh theo điểm cosine
        rel = (g["label"].to_numpy()[order] == c).astype(int)
        rows.append({"task": task, "query_class": c, "n_images": len(g),
                     "P@10": rel[:10].mean(), "P@50": rel[:50].mean(),
                     "AP": average_precision(rel), "chance": chance})
        if (task, c) in EXAMPLES:
            ex_data[(task, c)] = (g["image"].to_numpy()[order[:6]],
                                  t["texts"][m][0], (X @ q)[order[:6]])

res = pd.DataFrame(rows)
print("=== Truy xuất ảnh từ câu mô tả (không huấn luyện thêm) ===")
print("P@K = tỉ lệ ảnh đúng lớp trong K kết quả đầu; AP = độ chính xác trung bình; chance = mức ngẫu nhiên\n")
print(res.round(3).to_string(index=False))
summary = res.groupby("task")[["P@10", "P@50", "AP", "chance"]].mean().round(3)
print("\n=== Trung bình theo bài (mAP = trung bình AP) ===")
print(summary.rename(columns={"AP": "mAP"}).to_string())
res.round(4).to_csv("results/retrieval_metrics.csv", index=False)

fig, axes = plt.subplots(len(EXAMPLES), 6, figsize=(16, 2.9 * len(EXAMPLES)))
for r, key in enumerate(EXAMPLES):
    names, text, scores = ex_data[key]
    for c in range(6):
        ax = axes[r, c]
        ax.imshow(Image.open(f"data/images_640/{names[c]}"))
        ax.axis("off")
        ax.set_title(f"{scores[c]:.2f}", fontsize=9)
    axes[r, 0].text(0, -0.28, f"Truy vấn: {text}", transform=axes[r, 0].transAxes,
                    fontsize=10, ha="left")
fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.02, hspace=0.55, wspace=0.05)
fig.savefig("results/fig_retrieval_examples.png", dpi=120)
print("\nĐã lưu results/retrieval_metrics.csv, results/fig_retrieval_examples.png")
