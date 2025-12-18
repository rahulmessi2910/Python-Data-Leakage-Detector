import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split

# -----------------------------
# ARGUMENT HANDLING
# -----------------------------
if len(sys.argv) < 3:
    print("Usage: python data_leakage_detector.py <data.csv> <target_column>")
    sys.exit(1)

file_path = sys.argv[1]
target_col = sys.argv[2]

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(file_path)

if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")

print("\nDATASET LOADED SUCCESSFULLY")
print("Shape:", df.shape)

X = df.drop(columns=[target_col])
y = df[target_col]

# -----------------------------
# 1. TARGET LEAKAGE CHECK
# -----------------------------
print("\n[1] TARGET LEAKAGE CHECK")

numeric_cols = X.select_dtypes(include=np.number).columns
leakage_found = False

for col in numeric_cols:
    corr = X[col].corr(y)
    if abs(corr) > 0.9:
        print(f"⚠️  High correlation with target: {col} (corr = {corr:.3f})")
        leakage_found = True

if not leakage_found:
    print("✅ No strong target leakage detected.")

# -----------------------------
# 2. TRAIN–TEST CONTAMINATION
# -----------------------------
print("\n[2] TRAIN–TEST CONTAMINATION CHECK")

X_train, X_test, _, _ = train_test_split(
    X, y, test_size=0.3, random_state=42
)

overlap = pd.merge(X_train, X_test, how="inner")

if len(overlap) > 0:
    print("⚠️  Possible overlap between train and test sets detected.")
else:
    print("✅ No overlap between train and test sets.")

# -----------------------------
# 3. TIME-BASED LEAKAGE CHECK
# -----------------------------
print("\n[3] TIME-BASED LEAKAGE CHECK")

time_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]

if time_cols:
    print("⚠️  Time-related columns detected:", time_cols)
    print("⚠️  Ensure chronological train-test split is used.")
else:
    print("✅ No obvious time-based leakage detected.")

# -----------------------------
# 4. ID / GROUP LEAKAGE CHECK
# -----------------------------
print("\n[4] ID / GROUP LEAKAGE CHECK")

id_cols = [c for c in df.columns if "id" in c.lower()]

if id_cols:
    print("⚠️  ID-like columns detected:", id_cols)
    print("⚠️  These may cause memorisation instead of learning.")
else:
    print("✅ No ID-based leakage detected.")

# -----------------------------
# FINAL VERDICT
# -----------------------------
issues = leakage_found or len(overlap) > 0 or len(time_cols) > 0 or len(id_cols) > 0

print("\nFINAL VERDICT")
if issues:
    print("⚠️  Potential data leakage risks detected. Review before modeling.")
else:
    print("✅ Dataset appears safe from common data leakage issues.")
