# Tìm kiếm ngữ nghĩa trên video lái xe bằng Qwen3-VL-Embedding-2B

Bài tập lớn môn Thị giác máy tính. Sinh viên: Nguyễn Văn Anh.

Đề tài: tìm đoạn video camera hành trình (dashcam) bằng một câu tiếng Anh, ví dụ `a rainy road at night`, dựa trên mô hình embedding đa phương thức **Qwen3-VL-Embedding-2B** và công cụ mã nguồn mở [SentrySearch](https://github.com/ssrajadh/sentrysearch). Dự án gồm hai phần:

1. **Tìm kiếm video**: video được cắt thành đoạn 30 giây, mỗi đoạn thành một vector và lưu vào ChromaDB. Câu truy vấn thành vector trong cùng không gian, đoạn nào gần nhất thì được trả về (độ đo cosine).
2. **Phân loại ảnh** trên BDD100K theo 3 bài toán (thời tiết, loại cảnh, giờ trong ngày): so sánh *zero-shot* (so ảnh với câu mô tả của lớp) với *linear probe* (hồi quy logistic học trên embedding đã đóng băng).

Mô hình Qwen3-VL-Embedding-2B (giấy phép Apache 2.0) **không được huấn luyện lại**. Phần tự huấn luyện chỉ là bộ phân loại tuyến tính nhỏ đặt phía trên embedding.

## Kết quả chính

Độ chính xác trên tập test (90 ảnh mỗi lớp: 360 ảnh cho thời tiết, 270 ảnh cho mỗi bài còn lại). Cấu hình được chọn chỉ dựa trên F1 trung bình của tập validation, tập test chỉ dùng để báo cáo.

| Bài toán | Số lớp | Zero-shot | Linear probe |
|---|---|---|---|
| Thời tiết (weather) | 4 | 60,3% | 84,4% |
| Loại cảnh (scene) | 3 | 65,2% | 76,3% |
| Giờ trong ngày (timeofday) | 3 | 77,4% | 85,2% |

Tìm kiếm video (các clip Mixkit, mô tả trong báo cáo): bộ truy vấn gốc đúng 8/8 ở vị trí đầu; bộ truy vấn diễn đạt lại đúng 7/9 ở vị trí đầu và 9/9 trong top 3. Chi tiết, phân tích lỗi và thí nghiệm đổi tham số nằm trong báo cáo.

Số liệu gốc nằm trong `results/`: `metrics.csv` (zero-shot và probe C=1), `metrics_tuned.csv`, `metrics_classifiers.csv` (so sánh bộ phân loại), `retrieval_metrics.csv` (truy xuất ảnh), `video_eval.csv` và `video_eval_hard.csv` (truy xuất video).

## Cấu trúc thư mục

| Thư mục / tệp | Nội dung | Trong repo? |
|---|---|---|
| `scripts/` | 15 script đánh số theo thứ tự chạy (`00` đến `13`) | có |
| `artifacts/` | chia dữ liệu (`splits.csv`), embedding (`embeddings/*.npz`), bộ phân loại (`probe_*.joblib`), truy vấn và nguồn video | có |
| `results/` | bảng số liệu (CSV) và hình dùng trong báo cáo | có |
| `requirements.txt`, `requirements-analysis.txt` | thư viện cho hai môi trường (xem bên dưới) | có |
| `data/` | BDD100K tải về (~800 MB) và ảnh đã thu nhỏ (~230 MB) | không (tự tạo) |
| `demo_videos/` | video Mixkit để thử tìm kiếm | không (tự tải) |
| `sentrysearch/` | bản clone của SentrySearch | không (tự clone) |

## Cài đặt

Cần hai môi trường vì phần mô hình (PyTorch, Transformers) nặng và đòi Python 3.11 hoặc 3.12, còn phần phân tích thì nhẹ.

**Môi trường A: phân tích** (script 01 đến 04, 07 đến 10, 12):

```bash
pip install -r requirements-analysis.txt
```

**Môi trường B: mô hình** (script 05, 06, 11 và công cụ `sentrysearch` cho script 13). Cần Python 3.11 hoặc 3.12, [uv](https://docs.astral.sh/uv/) và ffmpeg (`brew install ffmpeg` trên macOS):

```bash
git clone https://github.com/ssrajadh/sentrysearch.git
cd sentrysearch && git checkout e99605e     # bản đã dùng trong bài
uv tool install --python 3.12 --with gradio --with pandas --with scikit-learn --with joblib --with pillow ".[local]"
cd ..
```

Lần chạy đầu sẽ tải mô hình (~4 GB) từ Hugging Face. Các script 05, 06, 11 phải chạy bằng Python của môi trường này:

```bash
~/.local/share/uv/tools/sentrysearch/bin/python scripts/05_extract_embeddings.py
```

`requirements.txt` là danh sách đầy đủ (đã `freeze`) của môi trường B trên máy tác giả (Apple M3 Pro, 32 GB, chạy bằng GPU Apple/MPS). Dòng `sentrysearch` trong tệp đã được chuyển thành ghi chú, vì SentrySearch được cài từ thư mục clone ở trên.

## Cách chạy từng bước

Chạy từ thư mục gốc của repo. Cột "Môi trường" là A hoặc B ở mục trên.

| Script | Môi trường | Việc làm | Đầu ra |
|---|---|---|---|
| `00_download_clips.sh` | bash, cần `curl` và `ffprobe` | tải các clip Mixkit | `demo_videos/` |
| `01_inspect.py` | A | xem cấu trúc bộ BDD100K trên Hugging Face | in ra màn hình |
| `02_count_classes.py` | A | đếm số ảnh theo từng nhãn | `results/class_counts.csv` |
| `03_make_splits.py` | A | chọn 600 ảnh mỗi lớp, chia 70/15/15, thu nhỏ về chiều rộng 640 px | `artifacts/splits.csv`, `data/images_640/` |
| `04_stats_figures.py` | A | vẽ hình thống kê dữ liệu | `results/fig_original_distribution.png`, `fig_split_counts.png`, `fig_samples.png` |
| `05_extract_embeddings.py` | B | embed 4.875 ảnh (~0,68 giây mỗi ảnh, ~55 phút); chạy lại sẽ tiếp tục phần còn dở | `artifacts/embeddings/image_embeddings_qwen2b_768.npz` |
| `06_text_embeddings.py` | B | embed 3 câu mô tả cho mỗi lớp | `artifacts/embeddings/text_prompts_qwen2b_768.npz` |
| `07_evaluate.py` | A | zero-shot và linear probe (C=1) | `results/metrics.csv`, `fig_confusion_*.png`, `artifacts/probe_*.joblib` |
| `07b_probe_tuning.py` | A | thử các giá trị C, chọn theo validation (chạy sau 07) | `results/metrics_tuned.csv`, `artifacts/probe_tuned_*.joblib` |
| `08_error_analysis.py` | A | phân tích ảnh bị đoán sai | `results/fig_errors_*.png` |
| `09_classifiers_and_curve.py` | A | so sánh hồi quy logistic, SVM, k láng giềng; đường cong học | `results/metrics_classifiers.csv`, `fig_learning_curve.png`, `artifacts/probe_best_*.joblib` |
| `10_retrieval_eval.py` | A | đánh giá truy xuất ảnh bằng câu chữ (P@10, P@50, AP) | `results/retrieval_metrics.csv`, `fig_retrieval_examples.png` |
| `11_demo_app.py` | B | demo Gradio: phân loại ảnh và tìm ảnh bằng câu chữ | giao diện web trên máy |
| `12_export_labels.py` | A | xuất đủ ba nhãn gốc của mọi ảnh | `artifacts/all_labels.csv` |
| `13_video_queries.py` | B (cần lệnh `sentrysearch`) | chạy loạt truy vấn video và chấm điểm | `results/video_eval.csv` |

**Dữ liệu cho script 02 đến 04 và 12**: bộ BDD100K phiên bản Hugging Face `dgural/bdd100k` (định dạng FiftyOne) cần nằm ở `data/bdd100k/` (có `samples.json`). Ví dụ:

```bash
hf download dgural/bdd100k --repo-type dataset --local-dir data/bdd100k
```

**Tìm kiếm video** (script 13): script `00` tải 11 clip; nếu báo cáo dùng thêm clip khác thì đặt chung vào `demo_videos/` trước khi lập chỉ mục. Sau đó lập chỉ mục và chạy truy vấn. Chỉ mục nằm trong cơ sở dữ liệu của SentrySearch trên máy, không nằm trong repo.

```bash
bash scripts/00_download_clips.sh
sentrysearch index demo_videos --model qwen2b
python3 scripts/13_video_queries.py                                            # bộ truy vấn gốc
python3 scripts/13_video_queries.py artifacts/video_queries_hard.csv results/video_eval_hard.csv   # bộ diễn đạt lại
sentrysearch search "a rainy road at night"                                    # thử tay một câu
```

**Chạy demo Gradio** (cần `data/images_640/` và mô hình đã tải):

```bash
~/.local/share/uv/tools/sentrysearch/bin/python scripts/11_demo_app.py
```

### Tái lập nhanh, không cần tải dữ liệu và không cần mô hình

Vì embedding đã có sẵn trong `artifacts/`, bạn có thể chạy lại phần phân loại mà không phải tải 800 MB dữ liệu hay đợi 55 phút embed ảnh. Chỉ cần môi trường A:

```bash
python scripts/07_evaluate.py
python scripts/07b_probe_tuning.py
python scripts/09_classifiers_and_curve.py
```

Các script này ghi đè các tệp `results/*.csv` và `artifacts/probe_*.joblib`. Các bộ phân loại `.joblib` là tệp pickle của scikit-learn, nên `requirements-analysis.txt` ghim `scikit-learn==1.9.1`.

## Dữ liệu và giấy phép

- Ảnh: BDD100K (Yu và cộng sự, CVPR 2020), lấy qua bản Hugging Face `dgural/bdd100k`. Ảnh gốc không được đưa vào repo. Điều khoản sử dụng xem mục "Nguồn dữ liệu và giấy phép" trong báo cáo.
- Video thử nghiệm: các clip Mixkit, **không** được đưa vào repo. Danh sách clip, đường dẫn trang gốc và ghi chú giấy phép nằm ở `artifacts/video_sources.csv`.
- Mô hình Qwen3-VL-Embedding-2B: giấy phép Apache 2.0.

## Hạn chế

- Tập test nhỏ (90 ảnh mỗi lớp), khoảng tin cậy 95% khoảng ±3,7 điểm (thời tiết), ±5,1 (loại cảnh), ±4,2 (giờ trong ngày). Chênh lệch dưới vài điểm giữa các cấu hình nằm trong sai số.
- Phần video chỉ có hơn chục clip, mang tính minh họa.
- Chưa đo riêng chất lượng khi cắt vector từ 2048 xuống 768 chiều (đây là cấu hình mặc định của SentrySearch).
- "Huấn luyện" ở đây là học bộ phân loại tuyến tính trên embedding đóng băng (linear probe), không phải tinh chỉnh mô hình.

## Tài liệu tham khảo

1. Li và cộng sự. *Qwen3-VL-Embedding*. arXiv:2601.04720, 2026.
2. Yu và cộng sự. *BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning*. CVPR 2020.
3. SentrySearch: https://github.com/ssrajadh/sentrysearch
