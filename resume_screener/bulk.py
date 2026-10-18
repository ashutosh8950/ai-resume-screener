"""Pure helpers for bulk resume upload validation and result export."""

import csv
from io import StringIO


MAX_BULK_FILES = 10
MAX_BULK_FILE_BYTES = 5 * 1024 * 1024
CSV_COLUMNS = (
    "rank",
    "file_name",
    "overall_pct",
    "semantic_pct",
    "skill_coverage_pct",
    "strong_count",
    "partial_count",
    "gaps_count",
    "top_missing_skills",
    "mode",
)


def check_upload_limits(
    files: list[tuple[str, int]],
    max_files: int = MAX_BULK_FILES,
    max_file_bytes: int = MAX_BULK_FILE_BYTES,
) -> tuple[list[int], list[str]]:
    """Return eligible file indices and messages for bulk upload limit violations."""
    messages = []
    if len(files) > max_files:
        messages.append(
            f"Only the first {max_files} files can be processed; "
            f"{len(files) - max_files} additional file(s) will be ignored."
        )

    eligible_indices = []
    for index, (name, size) in enumerate(files[:max_files]):
        if size > max_file_bytes:
            messages.append(
                f"{name} exceeds the {max_file_bytes // (1024 * 1024)} MB upload limit and will be skipped."
            )
        else:
            eligible_indices.append(index)
    return eligible_indices, messages


def rank_results(rows: list[dict]) -> list[dict]:
    """Sort result summaries by overall percentage and add 1-based ranks."""
    ordered = sorted(rows, key=lambda row: row["overall_pct"], reverse=True)
    return [{"rank": index, **row} for index, row in enumerate(ordered, start=1)]


def sanitize_csv_cell(value: object) -> str:
    """Neutralize leading characters that spreadsheet software treats as formulas."""
    text = "" if value is None else str(value)
    if text.startswith(("=", "+", "-", "@", "\t", "\r")):
        return f"'{text}"
    return text


def build_ranked_csv(rows: list[dict]) -> str:
    """Serialize ranked rows with ASCII headers and formula-safe cell values."""
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            column: sanitize_csv_cell(row.get(column, ""))
            for column in CSV_COLUMNS
        })
    return output.getvalue()