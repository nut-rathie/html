from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "titanic.csv"
PREDICTIONS_PATH = Path(__file__).resolve().parent / "titanic_predictions.csv"


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


def infer_fill_value(rows: Iterable[Dict[str, str]], column: str) -> float:
    values: List[float] = []
    for row in rows:
        raw = row[column].strip()
        if raw:
            try:
                values.append(float(raw))
            except ValueError:
                continue
    if not values:
        return 0.0
    return sum(values) / len(values)


def extract_features(rows: Sequence[Dict[str, str]]) -> Tuple[List[List[float]], List[int], List[str]]:
    age_default = infer_fill_value(rows, "Age")
    fare_default = infer_fill_value(rows, "Fare")

    embarked_categories = ["C", "Q", "S"]
    passenger_ids: List[str] = []
    features: List[List[float]] = []
    targets: List[int] = []

    for row in rows:
        passenger_ids.append(row["PassengerId"])
        targets.append(int(row["Survived"]))

        pclass = float(row["Pclass"]) if row["Pclass"] else 0.0
        sex = 1.0 if row["Sex"].strip().lower() == "female" else 0.0
        age = float(row["Age"]) if row["Age"].strip() else age_default
        sibsp = float(row["SibSp"]) if row["SibSp"] else 0.0
        parch = float(row["Parch"]) if row["Parch"] else 0.0
        fare = float(row["Fare"]) if row["Fare"].strip() else fare_default

        embarked_one_hot = [0.0, 0.0, 0.0]
        embark_value = row["Embarked"].strip().upper()
        if embark_value in embarked_categories:
            embarked_one_hot[embarked_categories.index(embark_value)] = 1.0

        feature_row = [
            1.0,  # bias term
            pclass,
            sex,
            age,
            sibsp,
            parch,
            fare,
            *embarked_one_hot,
        ]
        features.append(feature_row)

    return features, targets, passenger_ids


def standardize_features(features: List[List[float]]) -> List[List[float]]:
    if not features:
        return features

    num_features = len(features[0])

    # transpose features to compute statistics
    for index in range(num_features):
        column = [row[index] for row in features]
        if index == 0:
            continue

        mean = sum(column) / len(column)
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        std = math.sqrt(variance) if variance > 0 else 1.0

        for row in features:
            row[index] = (row[index] - mean) / std

    return features


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(value)
        return z / (1.0 + z)


def dot_product(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def train_logistic_regression(
    features: List[List[float]],
    targets: Sequence[int],
    learning_rate: float = 0.1,
    epochs: int = 800,
) -> List[float]:
    if not features:
        return []

    weights = [0.0 for _ in range(len(features[0]))]
    n = len(features)

    for _ in range(epochs):
        gradients = [0.0 for _ in weights]
        for row, target in zip(features, targets):
            prediction = sigmoid(dot_product(weights, row))
            error = prediction - target
            for index, value in enumerate(row):
                gradients[index] += error * value

        for index in range(len(weights)):
            weights[index] -= (learning_rate / n) * gradients[index]

    return weights


def predict_probabilities(features: Iterable[Sequence[float]], weights: Sequence[float]) -> List[float]:
    probabilities: List[float] = []
    for row in features:
        probabilities.append(sigmoid(dot_product(weights, row)))
    return probabilities


def evaluate(predictions: Sequence[float], targets: Sequence[int]) -> Tuple[float, float]:
    total = len(targets)
    if total == 0:
        return 0.0, 0.0

    correct = 0
    log_loss = 0.0
    for probability, target in zip(predictions, targets):
        predicted_label = 1 if probability >= 0.5 else 0
        if predicted_label == target:
            correct += 1

        # clamp probabilities to avoid log(0)
        prob = min(max(probability, 1e-12), 1 - 1e-12)
        if target == 1:
            log_loss -= math.log(prob)
        else:
            log_loss -= math.log(1 - prob)

    accuracy = correct / total
    average_log_loss = log_loss / total
    return accuracy, average_log_loss


def write_predictions(
    passenger_ids: Sequence[str],
    probabilities: Sequence[float],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["PassengerId", "PredictedProbability", "PredictedSurvived"])
        for passenger_id, probability in zip(passenger_ids, probabilities):
            writer.writerow(
                [
                    passenger_id,
                    f"{probability:.6f}",
                    1 if probability >= 0.5 else 0,
                ]
            )


def main() -> None:
    rows = load_rows(DATA_PATH)
    features, targets, passenger_ids = extract_features(rows)
    standardized_features = standardize_features(features)

    weights = train_logistic_regression(standardized_features, targets)
    probabilities = predict_probabilities(standardized_features, weights)

    accuracy, average_log_loss = evaluate(probabilities, targets)
    write_predictions(passenger_ids, probabilities, PREDICTIONS_PATH)

    print("Model training completed.")
    print(f"Training accuracy: {accuracy:.3f}")
    print(f"Average log loss: {average_log_loss:.3f}")
    print(
        f"Predictions written to {PREDICTIONS_PATH.relative_to(Path.cwd())}"
    )


if __name__ == "__main__":
    main()
