"""Demo: phân loại ảnh dashcam + tìm ảnh bằng câu mô tả (Qwen3-VL-Embedding-2B).

Chạy bằng Python của SentrySearch (có PyTorch):
  /Users/mac/.local/share/uv/tools/sentrysearch/bin/python scripts/11_demo_app.py
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # mô hình đã có trong máy, không cần gọi mạng

import joblib
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent          # thư mục cv-final
E = ROOT / "artifacts" / "embeddings"
IMG_DIR = ROOT / "data" / "images_640"
TASKS = ["weather", "scene", "timeofday"]

# ---- Nạp dữ liệu đã chuẩn bị ở các bước trước ----
_z = np.load(E / "image_embeddings_qwen2b_768.npz", allow_pickle=False)
NAMES = _z["names"].tolist()
VECS = _z["vectors"]                                    # (số ảnh, 768), đã chuẩn hóa độ dài = 1
META = {}                                               # ảnh -> {bài: nhãn gốc}
for r in pd.read_csv(ROOT / "artifacts" / "splits.csv").itertuples():
    META.setdefault(r.image, {})[r.task] = r.label       # nhãn của bài mà ảnh được chọn
_labels_file = ROOT / "artifacts" / "all_labels.csv"
if _labels_file.exists():                                 # nếu có: bổ sung đủ cả 3 nhãn gốc
    for r in pd.read_csv(_labels_file).itertuples():
        for t in TASKS:
            v = getattr(r, t)
            if isinstance(v, str):
                META.setdefault(r.image, {})[t] = v
PROBES = {t: joblib.load(ROOT / "artifacts" / f"probe_best_{t}.joblib") for t in TASKS}

_embedder = None


def get_embedder():
    """Nạp mô hình lần đầu khi cần (mất vài giây), các lần sau dùng lại."""
    global _embedder
    if _embedder is None:
        from sentrysearch.local_embedder import LocalEmbedder
        _embedder = LocalEmbedder(model_name="qwen2b", dimensions=768)
    return _embedder


def prep_image(img):
    """Tiền xử lý GIỐNG lúc huấn luyện: RGB, thu nhỏ chiều rộng 640, lưu JPEG chất lượng 90."""
    img = img.convert("RGB")
    w, h = img.size
    img = img.resize((640, round(h * 640 / w)), Image.LANCZOS)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    img.save(tmp.name, quality=90)
    return tmp.name


def classify_image(img):
    """Trả về xác suất 3 bài: weather, scene, timeofday."""
    if img is None:
        return {}, {}, {}
    path = prep_image(img)
    try:
        v = np.asarray(get_embedder().embed_image(path), dtype=np.float32)[None, :]
    finally:
        os.unlink(path)
    out = []
    for t in TASKS:
        obj = PROBES[t]
        proba = obj["model"].predict_proba(v)[0]
        out.append({c: float(p) for c, p in zip(obj["classes"], proba)})
    return tuple(out)


def search_images(query, k=8):
    """Tìm k ảnh có embedding gần nhất (cosine) với câu truy vấn."""
    query = (query or "").strip()
    if not query:
        return []
    q = np.asarray(get_embedder().embed_query(query), dtype=np.float32)
    scores = VECS @ q                                   # vector đã chuẩn hóa -> tích vô hướng = cosine
    top = np.argsort(-scores)[: int(k)]
    gallery = []
    for i in top:
        name = NAMES[i]
        m = META.get(name, {})
        caption = f"{scores[i]:.2f} | " + ", ".join(f"{t}: {m[t]}" for t in TASKS if t in m)
        gallery.append((str(IMG_DIR / name), caption))
    return gallery


def build_ui():
    import gradio as gr
    with gr.Blocks(title="Tìm kiếm ngữ nghĩa trên cảnh lái xe") as demo:
        gr.Markdown(
            "# Tìm kiếm ngữ nghĩa trên cảnh lái xe\n"
            "Mô hình: Qwen3-VL-Embedding-2B (chạy local). Dữ liệu: BDD100K (4.875 ảnh)."
        )
        with gr.Tab("Phân loại ảnh"):
            inp = gr.Image(type="pil", label="Tải ảnh dashcam lên")
            btn = gr.Button("Phân loại")
            with gr.Row():
                o1 = gr.Label(label="Thời tiết (weather)", num_top_classes=4)
                o2 = gr.Label(label="Loại cảnh (scene)", num_top_classes=3)
                o3 = gr.Label(label="Giờ trong ngày (timeofday)", num_top_classes=3)
            btn.click(classify_image, inp, [o1, o2, o3])
        with gr.Tab("Tìm ảnh bằng câu chữ"):
            q = gr.Textbox(label="Mô tả (nên viết tiếng Anh)", value="a rainy road at night")
            k = gr.Slider(4, 16, value=8, step=4, label="Số kết quả")
            b = gr.Button("Tìm")
            gal = gr.Gallery(columns=4, label="Kết quả (điểm cosine | nhãn gốc của ảnh)")
            b.click(search_images, [q, k], gal)
            q.submit(search_images, [q, k], gal)
    return demo


if __name__ == "__main__":
    build_ui().launch()
