from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "training"))

from pretokenize_uitars_sft import partition_rows, row_budget


def _row(rollout: str, images: int) -> tuple[int, dict]:
    return 0, {
        "rollout_dir": rollout,
        "messages": [{"role": "user", "content": [{"type": "image"}] * images + [{"type": "text", "text": "x"}]}],
    }


def test_partition_steps_balances_budget_across_workers():
    rows = []
    for rollout_idx in range(4):
        for step in range(3):
            rows.append((len(rows), _row(f"rollout-{rollout_idx}", images=step + 1)[1]))
    shards = partition_rows(rows, workers=2, mode="steps")
    assert [len(shard) for shard in shards] == [6, 6]
    budgets = [sum(row_budget(row) for _, row in shard) for shard in shards]
    assert max(budgets) - min(budgets) <= row_budget(rows[0][1])


def test_partition_rollouts_keeps_rollout_groups_intact():
    rows = [
        (0, _row("a", 1)[1]),
        (1, _row("a", 1)[1]),
        (2, _row("b", 3)[1]),
        (3, _row("b", 3)[1]),
    ]
    shards = partition_rows(rows, workers=2, mode="rollouts")
    rollout_sets = [{row["rollout_dir"] for _, row in shard} for shard in shards]
    assert all(len(rollout_set) == 1 for rollout_set in rollout_sets)
