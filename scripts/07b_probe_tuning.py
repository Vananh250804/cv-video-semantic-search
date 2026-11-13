import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

E = pathlib.Path("artifacts/embeddings")
z = np.load(E / "image_embeddings_qwen2b_768.npz", allow_pickle=False)
idx = {n: i for i, n in enumerate(z["names"].tolist())}
IMG = z["vectors"]
splits = pd.read_csv("artifacts/splits.csv")
splits = splits[splits["image"].isin(idx)]

# Các cấu hình thử. Chọn cấu hình tốt nhất CHỈ bằng tập val; tập test chỉ dùng để báo cáo.
CONFIGS = []
for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 10]:
    CONFIGS.append(("chuẩn hóa+LR", C,
                    lambda C=C: make_pipeline(StandardScaler(),
                                              LogisticRegression(C=C, max_iter=5000))))
for C in [1, 10, 100, 1000]:
    CONFIGS.append(("LR thô", C, lambda C=C: LogisticRegression(C=C, max_iter=5000)))

try:
    old = pd.read_csv("results/metrics.csv")
    old = old[(old.method == "linear probe") & (old.split == "test")].set_index("task")
except Exception:
    old = None

rows = []
for task, g in splits.groupby("task", sort=False):
    classes = sorted(g["label"].unique())
    cid = {c: i for i, c in enumerate(classes)}

    def get(split):
        d = g[g["split"] == split]
        return IMG[[idx[n] for n in d["image"]]], np.array([cid[l] for l in d["label"]])

    Xtr, ytr = get("train"); Xva, yva = get("val"); Xte, yte = get("test")

    print(f"\n=== {task}: macro F1 trên tập val theo từng cấu hình ===")
    best = None
    for name, C, make in CONFIGS:
        model = make().fit(Xtr, ytr)
        f1v = f1_score(yva, model.predict(Xva), average="macro")
        print(f"  {name:14s} C={C:<6}  val F1 = {f1v:.4f}")
        if best is None or f1v > best[0]:
            best = (f1v, name, C, model)
    f1v, name, C, model = best
    p = model.predict(Xte)
    acc, f1 = accuracy_score(yte, p), f1_score(yte, p, average="macro")
    old_acc = f"{old.loc[task, 'accuracy']:.4f}" if old is not None and task in old.index else "?"
    print(f"--> Chọn: {name}, C={C} | TEST accuracy={acc:.4f}, macro F1={f1:.4f} (trước đó: {old_acc})")
    joblib.dump({"model": model, "classes": classes}, f"artifacts/probe_tuned_{task}.joblib")
    rows.append({"task": task, "config": name, "C": C, "val_f1": round(f1v, 4),
                 "test_accuracy": round(acc, 4), "test_macro_f1": round(f1, 4)})

pd.DataFrame(rows).to_csv("results/metrics_tuned.csv", index=False)
print("\n=== TỔNG HỢP SAU KHI TINH CHỈNH ===")
print(pd.DataFrame(rows).to_string(index=False))
