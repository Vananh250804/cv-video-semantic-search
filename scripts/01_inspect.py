from datasets import load_dataset

# Tải dataset (khoảng 700 MB, lưu vào cache nên chỉ tải một lần)
ds = load_dataset("dgural/bdd100k")
print(ds)

split = list(ds.keys())[0]
d = ds[split]
print("Tên split:", split, "| số dòng:", len(d))
print("Các cột:", d.column_names)
print("Kiểu dữ liệu:", d.features)

# In một dòng mẫu, cắt ngắn để khỏi quá dài
row = d[0]
for k, v in row.items():
    print(k, "->", type(v).__name__, str(v)[:200])
