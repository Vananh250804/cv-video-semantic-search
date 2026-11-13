import pathlib
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

E = pathlib.Path("artifacts/embeddings")
z = np.load(E / "image_embeddings_qwen2b_768.npz", allow_pickle=False)
idx = {n: i for i, n in enumerate(z["names"].tolist())}
IMG = z["vectors"]
t = np.load(E / "text_prompts_qwen2b_768.npz", allow_pickle=False)

splits = pd.read_csv("artifacts/splits.csv")
n_all = len(splits)
splits = splits[splits["image"].isin(idx)]
print(f"Có embedding cho {len(splits)}/{n_all} lượt gán nhãn\n")

pathlib.Path("results").mkdir(exist_ok=True)
pathlib.Path("artifacts").mkdir(exist_ok=True)
rows = []


def add(task, method, split, y, p, extra=""):
    acc = accuracy_score(y, p)
    f1 = f1_score(y, p, average="macro")
    rows.append({"task": task, "method": method, "split": split,
                 "accuracy": round(acc, 4), "macro_f1": round(f1, 4), "note": extra})
    return acc, f1


def plot_cm(ax, y, p, classes, title):
    cm = confusion_matrix(y, p, labels=range(len(classes)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")


for task, g in splits.groupby("task", sort=False):
    classes = sorted(g["label"].unique())
    cid = {c: i for i, c in enumerate(classes)}

    def get(split):
        d = g[g["split"] == split]
        X = IMG[[idx[n] for n in d["image"]]]
        y = np.array([cid[l] for l in d["label"]])
        return X, y

    Xtr, ytr = get("train")
    Xva, yva = get("val")
    Xte, yte = get("test")

    # --- Zero-shot: so từng ảnh với vector đại diện của mỗi lớp ---
    C = []
    for c in classes:
        m = (t["tasks"] == task) & (t["labels"] == c)
        v = t["vectors"][m].mean(axis=0)
        C.append(v / np.linalg.norm(v))
    C = np.stack(C)
    p_zs_va = (Xva @ C.T).argmax(1)
    p_zs_te = (Xte @ C.T).argmax(1)
    add(task, "zero-shot", "val", yva, p_zs_va)
    add(task, "zero-shot", "test", yte, p_zs_te)

    # --- Linear probe: chọn C bằng tập val, đánh giá trên tập test ---
    best = None
    for Creg in [1, 10, 100, 1000]:
        clf = LogisticRegression(C=Creg, max_iter=3000)
        clf.fit(Xtr, ytr)
        f1v = f1_score(yva, clf.predict(Xva), average="macro")
        if best is None or f1v > best[0]:
            best = (f1v, Creg, clf)
    f1v, Creg, clf = best
    p_lp_va = clf.predict(Xva)
    p_lp_te = clf.predict(Xte)
    add(task, "linear probe", "val", yva, p_lp_va, f"C={Creg}")
    add(task, "linear probe", "test", yte, p_lp_te, f"C={Creg}")
    joblib.dump({"clf": clf, "classes": classes}, f"artifacts/probe_{task}.joblib")

    print(f"=== {task} (linear probe, C={Creg}) — báo cáo chi tiết trên tập test ===")
    print(classification_report(yte, p_lp_te, target_names=classes, digits=3))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    plot_cm(axes[0], yte, p_zs_te, classes, f"{task}: zero-shot (test)")
    plot_cm(axes[1], yte, p_lp_te, classes, f"{task}: linear probe (test)")
    fig.tight_layout()
    fig.savefig(f"results/fig_confusion_{task}.png", dpi=150)
    plt.close(fig)

res = pd.DataFrame(rows)
res.to_csv("results/metrics.csv", index=False)
print("=== TỔNG HỢP (accuracy, macro F1) ===")
print(res.to_string(index=False))
print("\nĐã lưu: results/metrics.csv, results/fig_confusion_*.png, artifacts/probe_*.joblib")
