
from datasets import load_dataset
from pathlib import Path

OUTPUT_DIR = Path("datasets/processed")
MAX_PER_CLASS = 100

for label in ["fresh", "rotten"]:
    (OUTPUT_DIR / label).mkdir(parents=True, exist_ok=True)

print("Loading FreshGuard dataset...")

dataset = load_dataset(
    "Project-AgML/fresh_rotten_fruit_classification",
    "raw",
    split="train",
    streaming=True
)

counts = {"fresh": 0, "rotten": 0}

for item in dataset:
    label = item["label"]
    label_name = "fresh" if label == 0 else "rotten"

    if counts[label_name] >= MAX_PER_CLASS:
        if all(count >= MAX_PER_CLASS for count in counts.values()):
            break
        continue

    image = item["image"].convert("RGB")
    fruit_type = item["fruit_type"].replace("/", "_")

    filename = f"{fruit_type}_{counts[label_name]}.jpg"
    save_path = OUTPUT_DIR / label_name / filename

    image.save(save_path, quality=90)
    counts[label_name] += 1

    print(f"{label_name}: {counts[label_name]}")

    if all(count >= MAX_PER_CLASS for count in counts.values()):
        break

print("\nDataset download completed!")
print(counts)
