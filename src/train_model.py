
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from PIL import Image
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "raw"
TRAIN_CSV = DATA_DIR / "train.csv"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (96, 96)


def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image)

    features = hog(
        image_array,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=-1
    )

    return features


def main():
    print("\nFRESHGUARD MODEL TRAINING")
    print("=" * 40)

    df = pd.read_csv(TRAIN_CSV)

    features = []
    labels = []

    print(f"Total labelled images: {len(df)}")

    for index, row in df.iterrows():
        image_name = str(row["Id"])
        label = str(row["Label"])

        matches = list(DATA_DIR.rglob(image_name))

        matches = [
            p for p in matches
            if ".cache" not in p.parts
        ]

        if not matches:
            print(f"Missing image: {image_name}")
            continue

        try:
            feature = extract_features(matches[0])
            features.append(feature)
            labels.append(label)

        except Exception as error:
            print(f"Skipping {image_name}: {error}")

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(df)}")

    X = np.asarray(features)
    y = np.asarray(labels)

    print("\nClass distribution:")
    print(pd.Series(y).value_counts())

    if len(X) == 0:
        raise RuntimeError("No training images found.")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=5,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42
        ))
    ])

    print("\nTraining model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    print("\nVALIDATION RESULTS")
    print("=" * 40)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_val, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_val,
        predictions,
        labels=sorted(np.unique(y))
    ))

    print("\nTraining final model on all labelled images...")
    model.fit(X, y)

    model_path = MODEL_DIR / "freshguard_model.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "classes": sorted(np.unique(y).tolist()),
        "image_size": list(IMAGE_SIZE),
        "training_images": len(X),
        "validation_accuracy": float(accuracy)
    }

    with open(MODEL_DIR / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    print("\nMODEL SAVED SUCCESSFULLY!")
    print(model_path)
    print("FreshGuard training completed.")


if __name__ == "__main__":
    main()
