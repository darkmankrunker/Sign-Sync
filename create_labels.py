# create_labels.py — run this ONCE after collecting all letters
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

CSV_FILE = "asl_data/landmarks.csv"

if not os.path.exists(CSV_FILE):
    print(f"ERROR: {CSV_FILE} not found!")
    input("Press Enter...")
    exit()

df = pd.read_csv(CSV_FILE)
print(f"Loaded {len(df)} samples")
print("Letters found:", sorted(df['label'].unique()))

le = LabelEncoder()
le.fit(df['label'])

joblib.dump(le, "labels.pkl")
print("\nlabels.pkl created successfully!")
print("Detected letters:", list(le.classes_))
print("You can now run train.py")
input("\nPress Enter to close...")