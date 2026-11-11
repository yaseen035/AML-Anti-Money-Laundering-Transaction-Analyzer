from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
files = list(ROOT.rglob("elliptic_txs_features.csv"))

if not files:
    raise FileNotFoundError("elliptic_txs_features.csv was not found.")

feature_file = files[0]
output_dir = ROOT / "outputs"
output_dir.mkdir(exist_ok=True)

chunk_size = 15000
count = 0
sum_x = None
sum_x2 = None
cross_sum = None
minimum = None
maximum = None
feature_count = None

print("Starting statistical analysis:", feature_file)

for number, chunk in enumerate(
    pd.read_csv(
        feature_file,
        header=None,
        dtype={0: str},
        chunksize=chunk_size
    ),
    start=1
):
    x = chunk.iloc[:, 2:].apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    x = x.to_numpy(dtype=np.float64)

    if feature_count is None:
        feature_count = x.shape[1]
        sum_x = np.zeros(feature_count, dtype=np.float64)
        sum_x2 = np.zeros(feature_count, dtype=np.float64)
        cross_sum = np.zeros(
            (feature_count, feature_count), dtype=np.float64
        )
        minimum = np.full(feature_count, np.inf)
        maximum = np.full(feature_count, -np.inf)

    sum_x += x.sum(axis=0)
    sum_x2 += np.square(x).sum(axis=0)
    cross_sum += x.T @ x
    minimum = np.minimum(minimum, x.min(axis=0))
    maximum = np.maximum(maximum, x.max(axis=0))
    count += len(x)

    print(f"Processed {count:,} transactions")

mean = sum_x / count
variance = np.maximum((sum_x2 / count) - np.square(mean), 0)
std = np.sqrt(variance)

covariance = (cross_sum / count) - np.outer(mean, mean)
denominator = np.outer(std, std)

correlation = np.zeros_like(covariance)
np.divide(
    covariance,
    denominator,
    out=correlation,
    where=denominator > 0
)

correlation = np.clip(correlation, -1, 1)

names = [f"feature_{i + 1}" for i in range(feature_count)]

statistics = pd.DataFrame({
    "feature": names,
    "mean": mean,
    "variance": variance,
    "standard_deviation": std,
    "minimum": minimum,
    "maximum": maximum
})

statistics.to_csv(
    output_dir / "feature_statistics.csv",
    index=False
)

pd.DataFrame(
    correlation,
    index=names,
    columns=names
).to_csv(output_dir / "feature_correlation.csv")

print("\nSTATISTICAL ANALYSIS COMPLETED")
print("Transactions analyzed:", f"{count:,}")
print("Numerical features:", feature_count)
print("Files saved:")
print(output_dir / "feature_statistics.csv")
print(output_dir / "feature_correlation.csv")
print("\nFirst 10 feature statistics:")
print(statistics.head(10).to_string(index=False))
