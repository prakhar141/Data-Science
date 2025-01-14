# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import dask.dataframe as dd

# Function to clean data
def clean_data(df):
    # Optimize data types to reduce memory usage
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")

    # Replace NaN with the median of each column
    df = df.fillna(df.median())

    # Replace infinities and cap extreme values
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().all():
            continue

        col_max = df[col][df[col] != np.inf].max()
        col_min = df[col][df[col] != -np.inf].min()
        df[col].replace([np.inf], col_max, inplace=True)
        df[col].replace([-np.inf], col_min, inplace=True)

        # Cap values at the 99th percentile
        valid_values = df[col][~np.isnan(df[col])]
        if not valid_values.empty:
            cap_value = np.percentile(valid_values, 99)
            df[col] = np.where(df[col] > cap_value, cap_value, df[col])

    # Replace any remaining NaN or infinite values with 0
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    return df

# Load data with Dask (for large datasets)
dev_data = pd.read_csv("Dev_data_to_be_shared.csv")  # Load entire data into memory as Pandas after chunked read
val_data = pd.read_csv("validation_data_to_be_shared.csv")

# Clean datasets
dev_data_cleaned = clean_data(dev_data)
val_data_cleaned = clean_data(val_data)

# Split development data into features (X) and target (y)
X = dev_data_cleaned.drop(columns=["bad_flag", "account_number"])
y = dev_data_cleaned["bad_flag"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Optimize memory usage by converting to float32
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

# Scale data in smaller chunks
scaler = StandardScaler()

# Scale training data
X_train_scaled = scaler.fit_transform(X_train)

# Scale test data
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_log = log_model.predict(X_test_scaled)
y_pred_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate Logistic Regression
print("Logistic Regression Metrics:")
print(classification_report(y_test, y_pred_log))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_prob_log))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_log)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Validation Data Predictions
val_X = val_data_cleaned.drop(columns=["account_number"])
val_X_scaled = scaler.transform(val_X.astype("float32"))  # Scale validation data
val_probabilities = log_model.predict_proba(val_X_scaled)[:, 1]

# Create final submission file
submission = val_data_cleaned[["account_number"]].copy()
submission["predicted_probability"] = val_probabilities
submission.to_csv("validation_predictions.csv", index=False)

print("Submission file created: validation_predictions.csv")
