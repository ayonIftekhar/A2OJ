import pandas as pd
import joblib
import numpy as np

# ---------- Step 1: Load your test feature set ----------
# IMPORTANT: This CSV must include an 'id' column and 20 descriptor features (same order as training)
test_df = pd.read_csv("raspd_test.csv")  # replace with your file
X_test = test_df.drop(columns=["id"])               # features only
ids = test_df["id"].tolist()

# ---------- Step 2: Load the scaler ----------
scaler = joblib.load("weights/weights/scalers.pkl")
X_test_scaled = scaler.transform(X_test)

# ---------- Step 3: Load all 6 eRF models and predict ----------
predictions = []

for fold in range(6):
    model_path = f"weights/Extremely Random Forest_fold_{fold}_shuffle_0.joblib"
    model = joblib.load(model_path)
    y_pred = model.predict(X_test_scaled)
    predictions.append(y_pred)

# ---------- Step 4: Average predictions across folds ----------
y_final = np.mean(predictions, axis=0)

# ---------- Step 5: Create output DataFrame ----------
output_df = pd.DataFrame({
    "id": ids,
    "Predicted": y_final
})

# ---------- Step 6: Save to CSV ----------
output_df.to_csv("raspdplus_predictions.csv", index=False)
print("✅ Saved predictions to 'raspdplus_ensemble_predictions.csv'")
