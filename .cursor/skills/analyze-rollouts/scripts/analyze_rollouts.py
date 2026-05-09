#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_score(text: str):
    text = text.strip()
    try:
        value = float(text)
    except ValueError:
        return text, text
    if value == 1.0:
        return "success", text
    if value == 0.0:
        return "fail", text
    return f"score={text}", text


def has_client_error(record: dict) -> bool:
    return (
        record.get("action") == "DONE"
        and "client error" in str(record.get("response", "")).lower()
    )


def load_rows(results_root: Path):
    rows = []
    for result_path in sorted(results_root.glob("**/result.txt")):
        traj_path = result_path.with_name("traj.jsonl")
        if not traj_path.exists():
            continue

        raw_score = result_path.read_text(encoding="utf-8").strip()
        outcome, _ = parse_score(raw_score)

        records = []
        for line in traj_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        start_timestamp = records[0].get("action_timestamp", "") if records else ""
        rows.append(
            {
                "sort_key": (start_timestamp, result_path.parent.name),
                "outcome": outcome,
                "steps": len(records),
                "task_id": result_path.parent.name,
                "client_error": any(has_client_error(record) for record in records),
            }
        )

    rows.sort(key=lambda row: row["sort_key"])
    for index, row in enumerate(rows, 1):
        row["#"] = index
    return rows


def render_table(rows, renumber=False):
    lines = [
        "| # | outcome | steps | task_id | client_error |",
        "|---|---------|-------|---------|--------------|",
    ]
    for index, row in enumerate(rows, 1):
        row_number = index if renumber else row["#"]
        lines.append(
            f"| {row_number} | {row['outcome']} | {row['steps']} | `{row['task_id']}` | "
            f"{'true' if row['client_error'] else 'false'} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    args = parser.parse_args()

    rows = load_rows(args.results_root)
    non_client_error_rows = [row for row in rows if not row["client_error"]]

    print("## All Runs")
    print()
    print(render_table(rows))
    print()
    print("## Runs Without Client Error")
    print()
    print(render_table(non_client_error_rows, renumber=True))


if __name__ == "__main__":
    main()
