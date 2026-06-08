#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from uitars_collator import PretokenizedUitarsDataset, make_pretokenized_collator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretokenized-dir", type=Path, required=True)
    args = parser.parse_args()

    rank = 0
    world_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    dataset = PretokenizedUitarsDataset(args.pretokenized_dir)
    assigned_shards = sum(1 for idx in range(len(dataset.shard_paths)) if idx % world_size == rank)
    expected_total = sum(dataset.shard_counts)

    record_count = 0
    collate_batches = 0
    collator = make_pretokenized_collator(pad_token_id=0)
    pending: list[dict] = []

    for item in dataset:
        record_count += 1
        pending.append(item)
        if len(pending) == 2:
            collator(pending)
            collate_batches += 1
            pending = []

    if pending:
        collator(pending)
        collate_batches += 1

    if dist.is_initialized():
        totals = torch.tensor(
            [record_count, collate_batches, len(dataset), assigned_shards],
            device=device,
            dtype=torch.long,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if rank == 0:
            print(
                f"records={int(totals[0])} collate_batches={int(totals[1])} "
                f"rank_len_sum={int(totals[2])} assigned_shards={int(totals[3])} "
                f"expected_records={expected_total}"
            )
            if int(totals[0]) != expected_total:
                print("record count mismatch", file=sys.stderr)
                sys.exit(1)
            if int(totals[2]) != expected_total:
                print("rank __len__ sum mismatch", file=sys.stderr)
                sys.exit(1)
        dist.barrier()
        dist.destroy_process_group()
        return

    print(
        f"records={record_count} collate_batches={collate_batches} "
        f"rank_len={len(dataset)} assigned_shards={assigned_shards} "
        f"expected_records={expected_total}"
    )
    if record_count != expected_total or len(dataset) != expected_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
