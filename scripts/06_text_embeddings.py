import numpy as np
from sentrysearch.local_embedder import LocalEmbedder

# Mỗi lớp có 3 câu mô tả (tiếng Anh, vì mô hình nhận tiếng Anh tốt nhất).
# Về sau ta lấy trung bình 3 vector để có một "vector đại diện" cho lớp.
PROMPTS = {
    ("weather", "clear"): [
        "a dashcam photo of a road on a clear sunny day with blue sky",
        "a driving scene in bright sunshine with no clouds",
        "a car view of a road under a clear sky",
    ],
    ("weather", "overcast"): [
        "a dashcam photo of a road on an overcast day with a grey cloudy sky",
        "a gloomy driving scene under heavy grey clouds",
        "a car view of a road in dull overcast weather",
    ],
    ("weather", "snowy"): [
        "a dashcam photo of a snowy road with snow on the ground",
        "a driving scene in winter with snow covering the street",
        "a car view of a road during a snowfall",
    ],
    ("weather", "rainy"): [
        "a dashcam photo of a rainy road with raindrops on the windshield",
        "a driving scene in the rain with a wet road surface",
        "a car view of a road during heavy rain",
    ],
    ("scene", "city street"): [
        "a dashcam photo of a city street with buildings and traffic",
        "a driving scene in a busy downtown area with many cars",
        "a car view of an urban street with intersections",
    ],
    ("scene", "highway"): [
        "a dashcam photo of a highway with multiple lanes and fast traffic",
        "a driving scene on a freeway with guardrails",
        "a car view of an open multi-lane expressway",
    ],
    ("scene", "residential"): [
        "a dashcam photo of a quiet residential street with houses",
        "a driving scene in a suburban neighborhood with parked cars and lawns",
        "a car view of a narrow street between family homes",
    ],
    ("timeofday", "daytime"): [
        "a dashcam photo taken in bright daylight",
        "a driving scene during the day with the sun up",
        "a car view of a road at midday",
    ],
    ("timeofday", "night"): [
        "a dashcam photo taken at night with a dark road and headlights",
        "a driving scene in the dark with street lamps and car lights",
        "a car view of a road at night",
    ],
    ("timeofday", "dawn/dusk"): [
        "a dashcam photo taken at dawn or dusk with an orange sunset sky",
        "a driving scene at sunset with low sun and long shadows",
        "a car view of a road during twilight",
    ],
}

emb = LocalEmbedder(model_name="qwen2b", dimensions=768)
tasks, labels, texts, vecs = [], [], [], []
for (task, label), prompts in PROMPTS.items():
    for p in prompts:
        v = np.asarray(emb.embed_query(p), dtype=np.float32)
        tasks.append(task)
        labels.append(label)
        texts.append(p)
        vecs.append(v)
        print(f"{task:10s} {label:12s} ok  ({p[:50]}...)")

np.savez(
    "artifacts/embeddings/text_prompts_qwen2b_768.npz",
    tasks=np.array(tasks),
    labels=np.array(labels),
    texts=np.array(texts),
    vectors=np.stack(vecs).astype(np.float32),
)
print(f"Đã lưu {len(vecs)} vector câu mô tả (10 lớp x 3 câu)")
