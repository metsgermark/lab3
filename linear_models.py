from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
#утилиты 

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Численно устойчивая сигмоида."""
    z = np.clip(z, -500, 500)
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def _bce(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            probabilities: np.ndarray) -> ClassificationMetrics:
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )

def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    X_train = np.asarray(data.train_features, dtype=np.float64)
    y_train = np.asarray(data.train_target, dtype=np.float64).ravel()
    X_test = np.asarray(data.test_features, dtype=np.float64)

    n_samples, n_features = X_train.shape

    # Инициализация весов нулями
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0

    lr = float(data.learning_rate)
    epochs = int(data.epochs)

    loss_history: list[float] = []

    for _ in range(epochs):
        # Прямой проход
        y_pred = X_train @ w + b
        error = y_pred - y_train

        # MSE и градиенты (пакетный GD)
        loss = _mse(y_train, y_pred)
        loss_history.append(loss)

        dw = (2.0 / n_samples) * (X_train.T @ error)
        db = (2.0 / n_samples) * np.sum(error)

        # Шаг градиентного спуска
        w = w - lr * dw
        b = b - lr * db

    predictions = X_test @ w + b

    return RegressionResult(
        weights=w,
        bias=float(b),
        loss_history=loss_history,
        predictions=predictions,
    )



def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    X_train = np.asarray(data.train_features, dtype=np.float64)
    y_train = np.asarray(data.train_target, dtype=np.float64).ravel()
    X_test = np.asarray(data.test_features, dtype=np.float64)

    n_samples, n_features = X_train.shape

    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0

    lr = float(data.learning_rate)
    epochs = int(data.epochs)

    loss_history: list[float] = []

    for _ in range(epochs):
        z = X_train @ w + b
        p = _sigmoid(z)

        loss = _bce(y_train, p)
        loss_history.append(loss)

        error = p - y_train
        dw = (1.0 / n_samples) * (X_train.T @ error)
        db = (1.0 / n_samples) * np.sum(error)

        w = w - lr * dw
        b = b - lr * db

    # Предсказания для теста
    z_test = X_test @ w + b
    probabilities = _sigmoid(z_test)
    predictions = (probabilities >= 0.5).astype(np.int64)

    return ClassificationResult(
        weights=w,
        bias=float(b),
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=predictions,
    )



def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    X_train = np.asarray(data.train_features, dtype=np.float64)
    y_train = np.asarray(data.train_target, dtype=np.float64).ravel()
    X_test = np.asarray(data.test_features, dtype=np.float64)
    y_test = np.asarray(data.test_target, dtype=np.float64).ravel()

    # --- Собственная логистическая регрессия ---
    own_result = train_logistic_classifier(data)
    own_metrics = _classification_metrics(y_test, own_result.predictions, own_result.probabilities)

    # --- Decision Tree ---
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    dt_proba = dt.predict_proba(X_test)[:, 1]
    dt_pred = dt.predict(X_test)
    dt_metrics = _classification_metrics(y_test, dt_pred, dt_proba)

    # --- KNN ---
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_proba = knn.predict_proba(X_test)[:, 1]
    knn_pred = knn.predict(X_test)
    knn_metrics = _classification_metrics(y_test, knn_pred, knn_proba)

    return BaselineComparisonResult(
        logistic_regression=own_metrics,
        decision_tree=dt_metrics,
        knn=knn_metrics,
    )
