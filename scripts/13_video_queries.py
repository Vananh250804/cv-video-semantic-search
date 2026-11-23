"""Chạy hàng loạt truy vấn video bằng SentrySearch và chấm điểm so với đáp án bạn đã ghi trước.

Dùng (chạy từ thư mục cv-final):
  python3 scripts/13_video_queries.py [file_truy_van.csv] [file_ket_qua.csv]
File artifacts/video_queries.csv có 4 cột: query, expected_file, start, end  (start/end dạng mm:ss hoặc số giây)
"""
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

QUERY_FILE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/video_queries.csv")
OUT_FILE = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/video_eval.csv")
TOP_N = 5
LINE = re.compile(r"#(\d+)\s+\[([0-9.]+)\]\s+(.+?)\s+@\s+(\d+):(\d+)-(\d+):(\d+)")


def to_sec(x):
    x = str(x).strip()
    if ":" in x:
        m, s = x.split(":")
        return int(m) * 60 + int(s)
    return int(float(x))


def run_query(query):
    exe = shutil.which("sentrysearch")
    if exe is None:
        sys.exit("Không tìm thấy lệnh 'sentrysearch' trong PATH.")
    cmd = [exe, "search", query, "-n", str(TOP_N), "--no-trim", "--threshold", "0"]
    p = subprocess.run(cmd, input="y\n", capture_output=True, text=True)
    text = p.stdout + "\n" + p.stderr
    results = []
    for m in LINE.finditer(text):
        rank, score, fname, m1, s1, m2, s2 = m.groups()
        results.append({"rank": int(rank), "score": float(score), "file": Path(fname).name,
                        "start": int(m1) * 60 + int(s1), "end": int(m2) * 60 + int(s2)})
    return results, text


def is_hit(r, exp_file, es, ee):
    """Đúng nếu cùng file và khoảng thời gian có giao nhau với đáp án."""
    return r["file"] == Path(exp_file).name and r["start"] < ee and r["end"] > es


if not QUERY_FILE.exists():
    sys.exit(f"Chưa có {QUERY_FILE}. Hãy tạo file với các cột: query, expected_file, start, end")

rows, n_top1, n_top3, n = [], 0, 0, 0
rr_sum = 0.0
with open(QUERY_FILE, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        q = row["query"].strip()
        es, ee = to_sec(row["start"]), to_sec(row["end"])
        results, raw = run_query(q)
        if not results:
            print(f"\n[CẢNH BÁO] Không đọc được kết quả cho: {q}\n--- đầu ra gốc ---\n{raw[-800:]}")
        hits = [is_hit(r, row["expected_file"], es, ee) for r in results]
        top1 = bool(hits and hits[0])
        top3 = any(hits[:3])
        rank = next((i + 1 for i, h in enumerate(hits) if h), 0)   # hạng của đáp án đúng (0 = không có trong top-N)
        n += 1
        n_top1 += top1
        n_top3 += top3
        rr_sum += (1 / rank) if rank else 0
        best = results[0] if results else {"file": "-", "start": 0, "end": 0, "score": 0}
        rows.append({"query": q, "expected": f"{row['expected_file']} {row['start']}-{row['end']}",
                     "top1": f"{best['file']} {best['start']//60:02d}:{best['start']%60:02d}",
                     "score": best["score"], "top1_ok": int(top1), "top3_ok": int(top3), "correct_rank": rank})
        print(f"{'ĐÚNG' if top1 else ('top3' if top3 else 'SAI ')} | {q[:55]:55s} | top1: {rows[-1]['top1']} ({best['score']:.2f}) | hạng đúng: {rank or '-'}")

OUT_FILE.parent.mkdir(exist_ok=True)
with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nSố truy vấn: {n} | Top-1 đúng: {n_top1}/{n} | Có trong top-3: {n_top3}/{n}")
print(f"MRR (hạng nghịch đảo trung bình): {rr_sum / n:.3f}")
print(f"Đã lưu {OUT_FILE}")
