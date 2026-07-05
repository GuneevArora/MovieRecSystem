from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

features = np.load(
    DATA_DIR / "movie_numerical.npy"
)

scaler = StandardScaler()

scaled = scaler.fit_transform(features)

np.save(
    DATA_DIR / "movie_numerical_scaled.npy",
    scaled.astype(np.float32)
)

joblib.dump(
    scaler,
    MODEL_DIR / "movie_feature_scaler.joblib"
)

print("Original:", features.shape)
print("Scaled:", scaled.shape)