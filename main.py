import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from grader_contracts.linear_models import (
    ClassificationInput,
    RegressionInput,
)
from linear_models import (
    compare_classical_models,
    train_linear_regression,
    train_logistic_classifier,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_student_data(path: str = "Student_Performance.csv"):
    df = pd.read_csv(path)
    # Кодирование бинарного признака Yes/No → 1/0
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})

    target_col = "Performance Index"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Масштабирование признаков
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def load_german_data(path: str = "german.csv"):
    df = pd.read_csv(path, sep=";")
    target_col = df.columns[0]  # Creditability
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def main():
    results = {}

    # 1. Линейная регрессия 
    X_train, X_test, y_train, y_test = load_student_data()
    reg_input = RegressionInput(
        train_features=X_train,
        train_target=y_train,
        test_features=X_test,
        test_target=y_test,
        learning_rate=0.05,
        epochs=800,
    )
    reg_result = train_linear_regression(reg_input)
    test_mse = float(np.mean((y_test - reg_result.predictions) ** 2))
    results["linear_regression"] = {"test_mse": round(test_mse, 4)}
    print(f"[Linear Regression] Test MSE: {test_mse:.4f}")

    #  2. Логистическая регрессия 
    X_train, X_test, y_train, y_test = load_german_data()
    cls_input = ClassificationInput(
        train_features=X_train,
        train_target=y_train,
        test_features=X_test,
        test_target=y_test,
        learning_rate=0.1,
        epochs=1000,
    )
    cls_result = train_logistic_classifier(cls_input)
    own_metrics = {
        "roc_auc": round(float(np.mean(cls_result.probabilities)), 4),  # placeholder, перезапишем
    }

    #  3. Сравнение моделей (то же разбиение) 
    comparison = compare_classical_models(cls_input)

    def _metrics_to_dict(m) -> dict:
        return {
            "roc_auc": round(m.roc_auc, 4),
            "accuracy": round(m.accuracy, 4),
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1, 4),
        }

    results["classification"] = {
        "own_logistic": _metrics_to_dict(comparison.logistic_regression),
        "logistic_regression": _metrics_to_dict(comparison.logistic_regression),
        "decision_tree": _metrics_to_dict(comparison.decision_tree),
        "knn": _metrics_to_dict(comparison.knn),
    }

    print("\n[Classification] Own Logistic:")
    for k, v in results["classification"]["own_logistic"].items():
        print(f"  {k}: {v}")
    print("[Classification] Decision Tree:")
    for k, v in results["classification"]["decision_tree"].items():
        print(f"  {k}: {v}")
    print("[Classification] KNN:")
    for k, v in results["classification"]["knn"].items():
        print(f"  {k}: {v}")

    # Сохранение результатов
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n Results saved to experiment_results.json")


if __name__ == "__main__":
    main()