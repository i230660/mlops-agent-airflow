"""Generate a small reproducible synthetic dataset for the demo: data/dataset.csv (then `dvc add data/dataset.csv`)."""
import sys

import pandas as pd
from sklearn.datasets import make_classification

noise = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02  # raise (e.g. 0.4) to inject a data-quality failure
X, y = make_classification(n_samples=1000, n_features=8, n_informative=5, flip_y=noise, random_state=7)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
df["target"] = y
df.to_csv("data/dataset.csv", index=False)
print("wrote data/dataset.csv", df.shape, "flip_y=", noise)
