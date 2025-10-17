from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "titanic.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "titanic_summary.md"


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


def survival_rate(rows: Iterable[Dict[str, str]]) -> float:
    total = 0
    survivors = 0
    for row in rows:
        total += 1
        survivors += int(row["Survived"])
    return (survivors / total) * 100 if total else 0.0


def group_survival_rates(rows: Iterable[Dict[str, str]], key: str) -> List[Tuple[str, int, float]]:
    grouped: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "survivors": 0})
    for row in rows:
        value = row[key] or "Unknown"
        grouped[value]["total"] += 1
        grouped[value]["survivors"] += int(row["Survived"])

    results: List[Tuple[str, int, float]] = []
    for value, counts in sorted(grouped.items(), key=lambda item: item[0]):
        rate = (counts["survivors"] / counts["total"]) * 100 if counts["total"] else 0.0
        results.append((value, counts["total"], rate))
    return results


def numeric_column(rows: Iterable[Dict[str, str]], column: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        cell = row[column].strip()
        if cell:
            try:
                values.append(float(cell))
            except ValueError:
                continue
    return values


def missing_counts(rows: Iterable[Dict[str, str]], columns: Iterable[str]) -> List[Tuple[str, int]]:
    counts: List[Tuple[str, int]] = []
    for column in columns:
        missing = sum(1 for row in rows if not row[column].strip())
        counts.append((column, missing))
    return counts


def format_rate(rate: float) -> str:
    return f"{rate:.1f}%"


def build_summary(rows: List[Dict[str, str]]) -> str:
    total = len(rows)
    overall_rate = format_rate(survival_rate(rows))
    ages = numeric_column(rows, "Age")
    fares = numeric_column(rows, "Fare")

    summary_lines = [
        "# Titanic Sample Dataset Summary",
        "",
        f"* Total passengers: **{total}**",
        f"* Overall survival rate: **{overall_rate}**",
    ]

    if ages:
        summary_lines.append(f"* Average age (excluding missing): **{mean(ages):.1f} years**")
    if fares:
        summary_lines.append(f"* Average fare: **£{mean(fares):.2f}**")

    summary_lines.append("\n## Survival Rate by Sex")
    summary_lines.append("")
    for sex, count, rate in group_survival_rates(rows, "Sex"):
        summary_lines.append(f"* {sex.title()} (n={count}): {format_rate(rate)}")

    summary_lines.append("\n## Survival Rate by Passenger Class")
    summary_lines.append("")
    for pclass, count, rate in group_survival_rates(rows, "Pclass"):
        summary_lines.append(f"* Class {pclass} (n={count}): {format_rate(rate)}")

    summary_lines.append("\n## Missing Values")
    summary_lines.append("")
    for column, missing in missing_counts(rows, ["Age", "Cabin", "Embarked"]):
        summary_lines.append(f"* {column}: {missing}")

    return "\n".join(summary_lines) + "\n"


def main() -> None:
    rows = load_rows(DATA_PATH)
    summary = build_summary(rows)
    OUTPUT_PATH.write_text(summary, encoding="utf-8")
    print(f"Summary written to {OUTPUT_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
