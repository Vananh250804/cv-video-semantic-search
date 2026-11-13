import pathlib
from collections import Counter
import joblib
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
splits = pd.read_csv("artifacts/splits.csv")
splits = splits[splits["image"].isin(idx)]

for task, g in splits.groupby("task", sort=False):
    obj = joblib.load(f"artifacts/probe_tuned_{task}.joblib")
    model, classes = obj["model"], obj["classes"]
    d = g[g["split"] == "test"].reset_index(drop=True)
    X = IMG[[idx[n] for n in d["image"]]]
    proba = model.predict_proba(X)
    pred = proba.argmax(1)
    true = np.array([classes.index(l) for l in d["label"]])
    wrong = np.where(pred != true)[0]

    print(f"\n=== {task}: {len(wrong)}/{len(d)} ảnh test bị đoán sai ===")
    pairs = Counter((classes[true[i]], classes[pred[i]]) for i in wrong)
    for (a, b), n in pairs.most_common(5):
        print(f"  thật '{a}' bị đoán thành '{b}': {n} ảnh")

    # 12 ảnh sai mà mô hình "tự tin" nhất: thường là ảnh nhãn đáng ngờ hoặc ảnh khó
    order = wrong[np.argsort(-proba[wrong, pred[wrong]])][:12]
    fig, axes = plt.subplots(3, 4, figsize=(14, 8.5))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), order):
        ax.imshow(Image.open(f"data/images_640/{d['image'][i]}"))
        ax.set_title(f"thật: {classes[true[i]]}\nđoán: {classes[pred[i]]} (p={proba[i, pred[i]]:.2f})",
                     fontsize=9)
    fig.suptitle(f"{task}: 12 ảnh test đoán sai với độ tự tin cao nhất")
    fig.tight_layout()
    fig.savefig(f"results/fig_errors_{task}.png", dpi=120)
    plt.close(fig)
print("\nĐã lưu results/fig_errors_*.png")
