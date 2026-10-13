"""Calibrate cosine thresholds for the selected embedding profile."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _selected_profile() -> str:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("lite", "full"),
        default=os.environ.get("EMBEDDING_PROFILE", "lite").strip().lower(),
    )
    args = parser.parse_args()
    os.environ["EMBEDDING_PROFILE"] = args.profile
    return args.profile


PROFILE = _selected_profile()

from resume_screener.config import EMBEDDING_MODEL, EMBEDDING_MODEL_LITE
from resume_screener.embeddings import (
    BGE_QUERY_PREFIX,
    _embed_texts,
    get_embedding_model,
)


CALIBRATION_PAIRS = [
    ("strong", "Django REST Framework development", "Built Django REST Framework APIs and serializers"),
    ("strong", "React frontend development", "Built reusable React components with TypeScript"),
    ("strong", "Linux server administration", "Administered Linux servers and automated tasks with Bash"),
    ("strong", "Python automation scripting", "Wrote Python scripts to automate data processing"),
    ("strong", "SQL query optimization", "Optimized PostgreSQL schemas and complex SQL queries"),
    ("strong", "Deploy applications on AWS", "Deployed containerized services to AWS using Docker"),
    ("strong", "Build REST APIs with FastAPI", "Designed and maintained production FastAPI endpoints"),
    ("strong", "Train machine learning models", "Trained classification models with scikit-learn and evaluated results"),
    ("strong", "Lead cross-functional engineering teams", "Led a cross-functional team through planning and product delivery"),
    ("partial", "Django REST Framework knowledge", "Built a Flask REST API for internal services"),
    ("partial", "React frontend development", "Developed Vue components for a customer dashboard"),
    ("partial", "AWS cloud deployments", "Deployed production applications to Microsoft Azure"),
    ("partial", "Python programming experience", "Implemented Java services and automated build jobs"),
    ("partial", "Kubernetes orchestration", "Packaged and ran services in Docker containers"),
    ("partial", "SQL database analysis", "Created Excel dashboards for operational reporting"),
    ("partial", "Build REST APIs", "Integrated third-party APIs and documented request flows"),
    ("partial", "Coach and mentor engineers", "Collaborated with peers during code reviews and team planning"),
    ("irrelevant", "Django REST Framework knowledge", "Bachelor of Engineering, 2022-2026"),
    ("irrelevant", "React frontend component development experience required", "Built a Flask REST API for internal services"),
    ("irrelevant", "Linux system administration experience required", "Prepared monthly financial forecasts and variance reports"),
    ("irrelevant", "Build web applications with React", "Resolved customer billing questions by phone and email"),
    ("irrelevant", "Optimize SQL queries", "Designed visual identity systems and print campaigns"),
    ("irrelevant", "AWS infrastructure deployment", "Coordinated classroom instruction and student assessments"),
    ("irrelevant", "Train machine learning models", "Managed catering schedules and food safety inspections"),
    ("irrelevant", "Lead engineering teams", "Operated forklifts and maintained warehouse inventory"),
    ("irrelevant", "Django REST Framework development experience required", "Introduction to Python programming"),
    ("irrelevant", "React frontend component development experience required", "Software engineer with experience building backend services"),
]


def suggest_threshold(lower: list[float], upper: list[float]) -> tuple[float, str]:
    lower_max = max(lower)
    upper_min = min(upper)
    if lower_max < upper_min:
        return (lower_max + upper_min) / 2, "clean separation"

    values = sorted(set(lower + upper))
    candidates = [(left + right) / 2 for left, right in zip(values, values[1:])]
    candidates.extend([values[0] - 1e-6, values[-1] + 1e-6])
    best_threshold = candidates[0]
    best_accuracy = -1.0
    for threshold in candidates:
        true_negative_rate = sum(value < threshold for value in lower) / len(lower)
        true_positive_rate = sum(value >= threshold for value in upper) / len(upper)
        balanced_accuracy = (true_negative_rate + true_positive_rate) / 2
        if balanced_accuracy > best_accuracy:
            best_accuracy = balanced_accuracy
            best_threshold = threshold
    return best_threshold, f"overlap; best balanced accuracy {best_accuracy:.0%}"


def main() -> None:
    model_name = EMBEDDING_MODEL_LITE if PROFILE == "lite" else EMBEDDING_MODEL
    model = get_embedding_model()
    query_prefix = BGE_QUERY_PREFIX if PROFILE == "full" else ""
    queries = [query_prefix + requirement for _, requirement, _ in CALIBRATION_PAIRS]
    passages = [evidence for _, _, evidence in CALIBRATION_PAIRS]
    vectors = _embed_texts(model, queries + passages)
    query_vectors = vectors[:len(queries)]
    passage_vectors = vectors[len(queries):]
    scores = np.diag(cosine_similarity(query_vectors, passage_vectors)).tolist()

    by_label = {label: [] for label in ("irrelevant", "partial", "strong")}
    print(f"Profile: {PROFILE} ({model_name})")
    print("LABEL      COSINE   REQUIREMENT  ->  RESUME EVIDENCE")
    print("-" * 112)
    scored_pairs = []
    for (label, requirement, evidence), score in zip(CALIBRATION_PAIRS, scores):
        by_label[label].append(float(score))
        scored_pairs.append((label, requirement, evidence, float(score)))
        print(f"{label:<10} {score:0.4f}   {requirement}  ->  {evidence}")

    partial_threshold, partial_note = suggest_threshold(
        by_label["irrelevant"], by_label["partial"]
    )
    strong_threshold, strong_note = suggest_threshold(
        by_label["partial"] + by_label["irrelevant"], by_label["strong"]
    )
    print("\nGROUP RANGES")
    for label in ("irrelevant", "partial", "strong"):
        values = by_label[label]
        print(f"{label:<10} {min(values):.4f} .. {max(values):.4f}")
    print(f"\nSuggested PARTIAL_MATCH_THRESHOLD: {partial_threshold:.4f} ({partial_note})")
    print(f"Suggested STRONG_MATCH_THRESHOLD:  {strong_threshold:.4f} ({strong_note})")
    partial_errors = [
        item for item in scored_pairs
        if (item[0] == "irrelevant" and item[3] >= partial_threshold)
        or (item[0] == "partial" and item[3] < partial_threshold)
    ]
    strong_errors = [
        item for item in scored_pairs
        if (item[0] == "strong" and item[3] < strong_threshold)
        or (item[0] != "strong" and item[3] >= strong_threshold)
    ]
    print(f"Partial/gap cutoff errors: {len(partial_errors)} of {len(scored_pairs)}")
    for label, requirement, evidence, score in partial_errors:
        print(f"  {label:<10} {score:.4f}  {requirement} -> {evidence}")
    print(f"Strong cutoff errors: {len(strong_errors)} of {len(scored_pairs)}")
    for label, requirement, evidence, score in strong_errors:
        print(f"  {label:<10} {score:.4f}  {requirement} -> {evidence}")
    print("Score scale: max(0, (cosine - partial threshold) / (1 - partial threshold))")
    del query_vectors, passage_vectors, vectors


if __name__ == "__main__":
    main()
