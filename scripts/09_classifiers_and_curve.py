import pathlib
import warnings
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)

E = pathlib.Path("artifacts/embeddings")
z = np.load(E / "image_embeddings_qwen2b_768.npz", allow_pickle=False)
idx = {n: i for i, n in enumerate(z["names"].tolist())}
IMG = z["vectors"]
splits = pd.read_csv("artifacts/splits.csv")
splits = splits[splits["image"].isin(idx)]
pathlib.Path("results").mkdir(exist_ok=True)

# ---- Danh sách bộ phân loại để so sánh ----
CANDS = []
for C in [0.1, 0.3, 1]:
    CANDS.append(("LR thô", f"C={C}",
                  lambda C=C: LogisticRegression(C=C, max_iter=5000)))
for C in [0.0001, 0.0003, 0.001]:
    CANDS.append(("LR chuẩn hóa", f"C={C}",
                  lambda C=C: make_pipeline(StandardScaler(),
                                            LogisticRegression(C=C, max_iter=5000))))
for C in [1, 3, 10, 30]:
    CANDS.append(("SVM RBF", f"C={C}",
                  lambda C=C: SVC(kernel="rbf", C=C, gamma="scale", probability=True,
                                  random_state=0)))
for k in [5, 15, 31]:
    CANDS.append(("kNN cosine", f"k={k}",
                  lambda k=k: KNeighborsClassifier(n_neighbors=k, metric="cosine",
                                                   algorithm="brute", weights="distance")))

rows = []
curves = {}
for task, g in splits.groupby("task", sort=False):
    classes = sorted(g["label"].unique())
    cid = {c: i for i, c in enumerate(classes)}

    def get(split):
        d = g[g["split"] == split]
        return IMG[[idx[n] for n in d["image"]]], np.array([cid[l] for l in d["label"]])

    Xtr, ytr = get("train"); Xva, yva = get("val"); Xte, yte = get("test")

    print(f"\n=== {task}: so sánh bộ phân loại (macro F1 trên val) ===")
    best = None
    for fam, par, make in CANDS:
        m = make().fit(Xtr, ytr)
        f1v = f1_score(yva, m.predict(Xva), average="macro")
        print(f"  {fam:14s} {par:10s} val F1 = {f1v:.4f}")
        if best is None or f1v > best[0]:
            best = (f1v, fam, par, m)
    f1v, fam, par, m = best
    p = m.predict(Xte)
    acc, f1 = accuracy_score(yte, p), f1_score(yte, p, average="macro")
    print(f"--> Chọn theo val: {fam} {par} | TEST accuracy={acc:.4f}, macro F1={f1:.4f}")
    joblib.dump({"model": m, "classes": classes}, f"artifacts/probe_best_{task}.joblib")
    rows.append({"task": task, "model": f"{fam} {par}", "val_f1": round(f1v, 4),
                 "test_accuracy": round(acc, 4), "test_macro_f1": round(f1, 4)})

    # ---- Đường cong học: càng nhiều dữ liệu train thì val F1 có tăng không? ----
    fr = [0.25, 0.5, 0.75, 1.0]
    mean, std = [], []
    for f in fr:
        sc = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            sel = rng.choice(len(Xtr), size=int(len(Xtr) * f), replace=False)
            mm = make_pipeline(StandardScaler(), LogisticRegression(C=0.001, max_iter=5000))
            mm.fit(Xtr[sel], ytr[sel])
            sc.append(f1_score(yva, mm.predict(Xva), average="macro"))
        mean.append(np.mean(sc)); std.append(np.std(sc))
    curves[task] = (fr, mean, std)
    print("  Đường cong học (LR chuẩn hóa, C=0.001): " +
          ", ".join(f"{int(f*100)}% dữ liệu -> {m_:.3f}" for f, m_ in zip(fr, mean)))

pd.DataFrame(rows).to_csv("results/metrics_classifiers.csv", index=False)
print("\n=== TỔNG HỢP ===")
print(pd.DataFrame(rows).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4.5))
for task, (fr, mean, std) in curves.items():
    ax.errorbar([f * 100 for f in fr], mean, yerr=std, marker="o", capsize=3, label=task)
ax.set_xlabel("% dữ liệu huấn luyện đã dùng")
ax.set_ylabel("Macro F1 trên tập val")
ax.set_title("Đường cong học (5 lần lấy mẫu ngẫu nhiên)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/fig_learning_curve.png", dpi=150)
print("Đã lưu results/fig_learning_curve.png, results/metrics_classifiers.csv")
