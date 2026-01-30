import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA_PATH = "telco-customer-churn.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Drop ID
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # One-hot encode
    X = pd.get_dummies(X, drop_first=True)
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score((y_test == "Yes").astype(int), y_proba)
    print("ROC-AUC:", auc)

    joblib.dump(model, "logistic_churn_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(feature_columns, "feature_columns.pkl")

    print("Saved model and artifacts")

if __name__ == "__main__":
    main()
