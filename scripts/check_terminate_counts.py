#!/usr/bin/env python
"""Check that each trajectory has at most one terminate action."""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
from typing import Any, Dict, List, Tuple

import pyarrow.parquet as pq


def _parse_traj(t: Any) -> Dict[str, Any]:
    return json.loads(t) if isinstance(t, str) else t


def _count_terminate_actions(traj: Dict[str, Any]) -> int:
    count = 0
    for step in traj.values():
        action_output = step.get("action", {}).get("action_output", {}) or {}
        if action_output.get("action_name") == "terminate":
            count += 1
    return count


def _check_file(fp: str, batch_size: int) -> Tuple[str, int, List[Tuple[str, int, int]]]:
    fname = os.path.basename(fp)
    pf = pq.ParquetFile(fp)
    row_idx = 0
    total = 0
    bad = []
    for batch in pf.iter_batches(
        batch_size=batch_size,
        columns=["trajectory"],
    ):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            total += 1
            traj = _parse_traj(row["trajectory"])
            n_terminate = _count_terminate_actions(traj)
            if n_terminate > 1:
                bad.append((fname, row_idx, n_terminate))
            row_idx += 1
    return fname, total, bad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/gpfs/scrubbed/reza/MolmoWeb-SyntheticTrajs/data_filtered",
        help="Directory containing parquet files with a trajectory column.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parquet files to check concurrently.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=50,
        help="Maximum violating rows to print.",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"No parquet files found under {args.data_dir}")

    bad: List[Tuple[str, int, int]] = []
    total = 0
    if args.num_workers <= 1:
        for fp in files:
            _, file_total, file_bad = _check_file(fp, args.batch_size)
            total += file_total
            bad.extend(file_bad)
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_workers
        ) as executor:
            future_to_file = {
                executor.submit(_check_file, fp, args.batch_size): fp
                for fp in files
            }
            for fut in concurrent.futures.as_completed(future_to_file):
                _, file_total, file_bad = fut.result()
                total += file_total
                bad.extend(file_bad)

    print(f"Checked {total} trajectories across {len(files)} parquet file(s)")
    print(f"Found {len(bad)} trajectories with >1 terminate actions")

    for fname, row_idx, n_terminate in bad[: args.max_examples]:
        print(f"{fname} row={row_idx} terminate_count={n_terminate}")

    if len(bad) > args.max_examples:
        print(f"... plus {len(bad) - args.max_examples} more")

    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
