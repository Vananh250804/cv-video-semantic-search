import argparse, csv, os, sys, time
from pathlib import Path
import numpy as np
from sentrysearch.local_embedder import LocalEmbedder

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None,
                help="chỉ embed tối đa N ảnh mới trong lần chạy này (để đo tốc độ)")
ap.add_argument("--save-every", type=int, default=200)
args = ap.parse_args()

IMG_DIR = Path("data/images_640")
OUT = Path("artifacts/embeddings"); OUT.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT / "image_embeddings_qwen2b_768.npz"

with open("artifacts/splits.csv", newline="") as f:
    names = sorted({row["image"] for row in csv.DictReader(f)})
print(f"Tổng số ảnh cần embed: {len(names)}")

done = {}
if OUT_FILE.exists():
    z = np.load(OUT_FILE, allow_pickle=False)
    done = dict(zip(z["names"].tolist(), z["vectors"]))
    print(f"Đã có sẵn {len(done)} embedding, sẽ chạy tiếp phần còn lại")

def save():
    if not done:
        return
    keys = sorted(done)
    tmp = OUT_FILE.with_suffix(".tmp.npz")
    np.savez(tmp, names=np.array(keys),
             vectors=np.stack([done[k] for k in keys]).astype(np.float32))
    os.replace(tmp, OUT_FILE)

todo = [n for n in names if n not in done]
if args.limit:
    todo = todo[:args.limit]
print(f"Lần chạy này xử lý: {len(todo)} ảnh")

emb = LocalEmbedder(model_name="qwen2b", dimensions=768)
t0 = t_first = None
new = 0
try:
    for i, name in enumerate(todo, 1):
        try:
            v = np.asarray(emb.embed_image(str(IMG_DIR / name)), dtype=np.float32)
        except Exception as e:
            print("LỖI", name, repr(e), file=sys.stderr)
            with open(OUT / "failed.txt", "a") as ff:
                ff.write(name + "\n")
            continue
        done[name] = v
        new += 1
        if t_first is None:                      # bỏ qua ảnh đầu (đã gồm thời gian nạp mô hình)
            t_first = time.monotonic()
            print(f"Vector: shape={v.shape}, độ dài={np.linalg.norm(v):.3f}")
        if i % 25 == 0 or i == len(todo):
            per = (time.monotonic() - t_first) / max(new - 1, 1)
            remain = (len(names) - len(done)) * per / 60
            print(f"[{i}/{len(todo)}] {per:.2f} giây/ảnh | còn khoảng {remain:.0f} phút cho toàn bộ")
        if new % args.save_every == 0:
            save()
finally:
    save()
    print(f"Đã lưu {len(done)} embedding vào {OUT_FILE}")
