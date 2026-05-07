"""Offline trajectory filter for FARA SFT data.

Drops blank-prefix and trailing-duplicate (image, step) pairs from each
trajectory, mirroring the runtime behavior of
`Qwen2VLImageProcessor._compute_image_status`:
    blank_first / blank_drop  → drop the (image, step) pair
    trailing_dup              → drop the (image, step) pair
    active / mid_dup          → keep as-is (mid_dup is multiscale-only and
                                stays at runtime in the image processor)

After running this once, point train_fara.py at --data_path <output_dir>.
With pre-cleaned data, only 'active' and 'mid_dup' statuses remain at
runtime — the blank/trailing branches in
`_compute_image_status` / `_apply_image_status` can then be deleted.

Schema preserved: every input column passes through unchanged except
`images` and `trajectory`, which are filtered in lockstep (one image per
trajectory step).

Usage:
    python scripts/preprocess_trajectories.py \\
        --data_path /gpfs/scrubbed/reza/MolmoWeb-SyntheticTrajs/data \\
        --output_dir /gpfs/scrubbed/reza/MolmoWeb-SyntheticTrajs/data_filtered \\
        --num_proc 8
"""
import argparse
import io
from pathlib import Path
from typing import Any, Dict, List
import json

import numpy as np
from datasets import load_dataset
from PIL import Image


def _to_array(img_entry: Any) -> np.ndarray:
    """Decode any supported image entry to a uint8 HWC numpy array."""
    if isinstance(img_entry, dict) and img_entry.get("bytes") is not None:
        return np.asarray(
            Image.open(io.BytesIO(img_entry["bytes"])).convert("RGB"),
            dtype=np.uint8,
        )
    if isinstance(img_entry, (bytes, bytearray)):
        return np.asarray(
            Image.open(io.BytesIO(bytes(img_entry))).convert("RGB"),
            dtype=np.uint8,
        )
    if isinstance(img_entry, Image.Image):
        return np.asarray(img_entry.convert("RGB"), dtype=np.uint8)
    if isinstance(img_entry, np.ndarray):
        return img_entry if img_entry.dtype == np.uint8 else img_entry.astype(np.uint8)
    raise TypeError(f"unsupported image type: {type(img_entry)}")


def _get_action_name(step: Any) -> str:
    """Best-effort extraction of the lowercase action name from a step dict.
    Mirrors the nested-fallback chain in train_fara._format_assistant_message.
    """
    if not isinstance(step, dict):
        return ""
    # Nested: step["action"]["action_output"]["action_name"|"action"]
    action_block = step.get("action") if isinstance(step.get("action"), dict) else None
    if action_block is not None:
        ao = action_block.get("action_output") if isinstance(action_block.get("action_output"), dict) else {}
        if isinstance(ao, dict):
            for key in ("action_name", "action"):
                v = ao.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip().lower()
    # Flat fallbacks.
    for key in ("action_name", "action"):
        v = step.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    args = step.get("arguments")
    if isinstance(args, dict) and isinstance(args.get("action"), str):
        return args["action"].strip().lower()
    return ""


def _first_send_msg_idx(trajectory: List[Any]) -> int:
    """Return the index of the first send_msg_to_user step, or -1 if none."""
    for i in sorted(trajectory.keys()):
        step = trajectory[i]
        if _get_action_name(step) == "send_msg_to_user":
            return i
    return -1


def _filter_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Filter a trajectory in three passes:

    1. Truncate at the first `send_msg_to_user` step (keep that step; drop
       everything after it — those are post-final-answer artifacts).
    2. Drop leading-blank (image, step) pairs.
    3. Drop the trailing pair if its screenshot pixel-equals the previous
       active one — skipped when pass 1 truncated, since the truncated tail
       is the send_msg_to_user step and must always be kept.

    Lazy decoding: only the leading run (until first non-blank) and the last
    two active frames are PNG-decoded.
    """
    images = list(row.get("images") or [])
    trajectory = row.get("trajectory")
    if isinstance(trajectory, str):
        trajectory = json.loads(trajectory)
    trajectory = {int(k): v for k, v in (trajectory or {}).items()}  # ensure int keys
    # n = min(len(images), len(trajectory))
    assert len(images) == len(trajectory), f"mismatched lengths: {len(images)} images vs {len(trajectory)} trajectory steps"
    if len(images) == 0:
        return {"images": images, "trajectory": trajectory}

    # Pass 1: truncate at first send_msg_to_user.
    # Trajectory is 1-indexed (keys 1..N); images are 0-indexed (0..N-1).
    # `images[i]` aligns with `trajectory[i + 1]`.
    send_idx = _first_send_msg_idx(trajectory)
    truncated = send_idx != -1
    if truncated:
        images = images[:send_idx]
        trajectory = {idx: step for idx, step in trajectory.items() if idx <= send_idx}

    # n is the number of (image, step) pairs we'll filter; both lists have len n.
    n = len(images)
    keep = [True] * n

    # Pass 2: drop the very first frame if blank. Subsequent leading blanks
    # (e.g. images[1] also blank) are kept — the row instruction lives in a
    # separate column and is re-attached to whichever step ends up at i=0
    # at message-build time, so the task signal is preserved either way.
    if n > 0 and float(_to_array(images[0]).astype(np.float32).var()) == 0.0:
        keep[0] = False

    # Pass 3: trailing dup — last active frame pixel-equal to the prior active.
    # Skipped after a send_msg truncation: the truncated tail is the model's
    # final answer step and must always be kept.
    if not truncated:
        active = [i for i, k in enumerate(keep) if k]
        if len(active) >= 2:
            last = _to_array(images[active[-1]])
            prev = _to_array(images[active[-2]])
            if last.shape == prev.shape and np.array_equal(last, prev):
                keep[active[-1]] = False

    # Output: images stay 0-indexed list; trajectory re-keyed to 1-indexed dict
    # so kept steps are contiguous (1, 2, … k), then JSON-serialized so the
    # output column has a single fixed string type. Without this, pyarrow
    # tries to infer a struct schema per batch and fails when different
    # batches contain different action types (click/scroll/send_msg/…) with
    # different argument fields. train_fara._parse_trajectory already accepts
    # JSON strings.
    kept = [i for i in range(n) if keep[i]]
    new_images = [images[i] for i in kept]
    new_trajectory = {str(new_idx + 1): trajectory[i + 1] for new_idx, i in enumerate(kept)}
    return {"images": new_images, "trajectory": json.dumps(new_trajectory)}


def _last_three_identical(arrays: List[np.ndarray]) -> bool:
    """True if the last 3 frames in `arrays` are pixel-identical."""
    if len(arrays) < 3:
        return False
    a, b, c = arrays[-3], arrays[-2], arrays[-1]
    return (
        a.shape == b.shape == c.shape
        and np.array_equal(a, b)
        and np.array_equal(b, c)
    )


def _scan_triple_dup(in_path: Path, num_proc: int, max_hits: int) -> None:
    """Walk rows once and report trajectories whose last 3 frames are identical.
    Sets a `breakpoint()` on each hit so the trajectory can be inspected in
    the debugger. Use --max_hits 1 to stop at the first one.
    """
    ds = load_dataset(
        "parquet", data_files=str(in_path), split="train", num_proc=num_proc,
    )
    print(f"[scan] {in_path.name}: {len(ds)} rows")
    hits = 0
    for idx, row in enumerate(ds):
        images = row.get("images") or []
        trajectory = row.get("trajectory") or []
        n = min(len(images), len(trajectory))
        if n < 3:
            continue
        arrays = [_to_array(img) for img in images[:n]]
        if _last_three_identical(arrays):
            sample_id = row.get("sample_id")
            print(f"[hit] row={idx} sample_id={sample_id} n_frames={n}")
            # breakpoint()  # inspect: row, arrays, trajectory
            hits += 1
            if hits >= max_hits:
                break
    print(f"[scan] done: {hits} hit(s)")


def _process_file(
    in_path: Path,
    out_path: Path,
    num_proc: int,
    writer_batch_size: int,
    overwrite: bool,
) -> None:
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path.name}: exists (use --overwrite to redo)")
        return

    ds = load_dataset(
        "parquet", data_files=str(in_path), split="train", num_proc=num_proc,
    )
    n_in = len(ds)

    ds = ds.map(
        _filter_row,
        num_proc=num_proc,
        writer_batch_size=writer_batch_size,
        desc=f"filter {in_path.name}",
    )
    ds = ds.filter(
        lambda r: bool(r["images"]),
        num_proc=num_proc,
        writer_batch_size=writer_batch_size,
        desc=f"drop empty {in_path.name}",
    )
    n_out = len(ds)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(out_path))
    print(f"[done] {in_path.name}: rows {n_in} → {n_out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True,
                   help="Input parquet file or directory of parquets")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write filtered parquets")
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--writer_batch_size", type=int, default=100,
                   help="Smaller values avoid pyarrow int32 offset overflow on "
                        "rows with large image payloads.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--check_triple_dup", action="store_true",
                   help="Scan-only: report trajectories whose last 3 frames are "
                        "pixel-identical and breakpoint() on each. No output written.")
    p.add_argument("--max_hits", type=int, default=1,
                   help="Stop after this many --check_triple_dup hits.")
    args = p.parse_args()

    in_path = Path(args.data_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: List[Path] = (
        [in_path] if in_path.is_file()
        else sorted(in_path.glob("*.parquet"))
    )
    if not files:
        raise SystemExit(f"no parquet files under {in_path}")

    print(f"processing {len(files)} file(s) → {out_dir}")
    for f in files:
        _process_file(
            f, out_dir / f.name,
            num_proc=args.num_proc,
            writer_batch_size=args.writer_batch_size,
            overwrite=args.overwrite,
        )
    
    if args.check_triple_dup:
        for f in files:
            _scan_triple_dup(f, num_proc=args.num_proc, max_hits=args.max_hits)
        return


if __name__ == "__main__":
    main()
