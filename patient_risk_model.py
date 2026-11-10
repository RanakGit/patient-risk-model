"""Reusable analysis functions and command-line runner for the vital signs model."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = ["age", "bp", "heart_rate", "Temp"]
CLUSTER_FEATURES = ["bp", "heart_rate", "Temp", "age"]


def analyze_vitals(data: pd.DataFrame):
    """Cluster records and fit the demonstration classifier on valid vital signs."""
    df = data.copy()
    df.columns = [str(column).strip().replace(" ", "_") for column in df.columns]

    if df.columns.duplicated().any():
        raise ValueError("Column names must be unique after trimming and normalization.")

    missing_columns = [column for column in FEATURES if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(missing_columns)
        )

    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    original_rows = len(df)
    df = df.dropna(subset=FEATURES).copy()
    dropped_rows = original_rows - len(df)

    if len(df) < 2:
        raise ValueError(
            "At least two rows with valid age, bp, heart_rate, and Temp values are required."
        )

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[CLUSTER_FEATURES])
    clusters = KMeans(n_clusters=2, n_init=10, random_state=42)
    df["Cluster"] = clusters.fit_predict(scaled_features)

    cluster_summary = df.groupby("Cluster")[CLUSTER_FEATURES].mean()

    df["RiskLevel"] = np.where(
        (df["Temp"] > 37) | (df["heart_rate"] > 85), "High", "Low"
    )
    labels = (df["RiskLevel"] == "High").astype(int)

    evaluation = None
    label_counts = labels.value_counts()
    if len(label_counts) == 2 and label_counts.min() >= 2 and len(df) >= 8:
        train_x, test_x, train_y, test_y = train_test_split(
            df[FEATURES],
            labels,
            test_size=0.25,
            random_state=42,
            stratify=labels,
        )
        evaluation_model = RandomForestClassifier(random_state=42)
        evaluation_model.fit(train_x, train_y)
        test_predictions = evaluation_model.predict(test_x)
        majority_label = train_y.value_counts().idxmax()
        evaluation = {
            "accuracy": accuracy_score(test_y, test_predictions),
            "test_size": len(test_y),
            "baseline_accuracy": accuracy_score(
                test_y, np.full(len(test_y), majority_label)
            ),
            "confusion_matrix": confusion_matrix(
                test_y, test_predictions, labels=[0, 1]
            ),
        }

    model = RandomForestClassifier(random_state=42)
    model.fit(df[FEATURES], labels)

    feature_importance = pd.Series(
        model.feature_importances_, index=FEATURES, name="Importance"
    ).sort_values()

    return df, cluster_summary, model, feature_importance, dropped_rows, evaluation


def main():
    project_dir = Path(__file__).resolve().parent
    input_path = project_dir / "vital_signs_clean.csv"
    if not input_path.is_file():
        raise FileNotFoundError(f"Cleaned CSV not found: {input_path}")

    source = pd.read_csv(input_path)
    print("Dataset Preview:")
    print(source.head(), "\n")
    print("Dataset information:")
    source.info()

    (
        df,
        cluster_summary,
        model,
        feature_importance,
        dropped_rows,
        evaluation,
    ) = analyze_vitals(source)
    print(f"\nRows excluded for missing or invalid values: {dropped_rows}")
    print("\nCluster Summary:\n", cluster_summary)
    print("\nRisk level counts:\n", df["RiskLevel"].value_counts())
    if evaluation is not None:
        print(
            "\nHoldout agreement with rule-generated labels:",
            f"{evaluation['accuracy']:.1%}",
            f"({evaluation['test_size']} holdout rows)",
        )
    else:
        print("\nHoldout evaluation unavailable: dataset is too small or has one class.")

    new_patient = pd.DataFrame(
        [[72, 160, 105, 38.5]],
        columns=FEATURES,
    )
    prediction = model.predict(new_patient)[0]
    risk_label = "High Risk" if prediction == 1 else "Low Risk"
    print(f"\nExample patient prediction: {risk_label}")

    output_path = project_dir / "Patient_Risk_Model_Output.xlsx"
    df.to_excel(output_path, index=False, engine="xlsxwriter")
    print(f"\nUpdated dataset saved at:\n{output_path}")
    print("\nFeature importance:\n", feature_importance)


if __name__ == "__main__":
    main()
