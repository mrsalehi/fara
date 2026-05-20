#!/usr/bin/env python
"""Full-corpus click->keyboard_type merge pipeline.

Phase 1 (judge): walk every click->keyboard_type pair in every parquet
under --data_dir, send each to the configured LLM judge, append the
decision to --decisions_log. Resumable: pairs whose key is already in the
log are skipped, so a crashed/aborted run just continues.

Phase 2 (apply): read the completed log, for each parquet row that has at
least one merge=true decision, rewrite the trajectory in-place (drop the
click step, convert the following keyboard_type step to canonical type,
fold the click's bbox + the LLM's merged_thought into it) and drop the
matching screenshot from the parallel `images` array. Pass-through rows
are copied untouched. Output: --out_dir/<same-filename>.parquet for every
input parquet.

By default both phases run. Use --skip_judge to apply existing decisions
without further LLM calls, or --skip_apply to only produce the log.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import glob
import json
import os
import shutil
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Reuse helpers (pair finder, action extractor, judge call, system prompt).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eyeball_click_type_pairs import (  # noqa: E402
    _find_pairs,
    _judge,
    _parse_traj,
)


def _pair_key(file: str, ri: int, ck: str, tk: str) -> str:
    return f"{file}#{ri}#{ck}#{tk}"


def _load_decisions_log(path: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "pair_key" in rec:
                out[rec["pair_key"]] = rec
    return out


def _make_client(provider: str):
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()
    if provider == "google":
        from google import genai
        return genai.Client()
    raise ValueError(f"unknown provider: {provider}")


_thread_local = threading.local()


def _get_thread_client(provider: str):
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = _make_client(provider)
        _thread_local.client = client
    return client


def _judge_pair(args: argparse.Namespace, fname: str, ri: int, ck: str, tk: str,
                click: dict, typ: dict) -> dict:
    client = _get_thread_client(args.provider)
    dec = _judge(client, args.model, args.provider, click, typ)
    return {
        "pair_key": _pair_key(fname, ri, ck, tk),
        "file": fname,
        "traj_idx": int(ri),
        "click_step_key": str(ck),
        "type_step_key": str(tk),
        "click_bbox": click["args"].get("bbox") or click["args"].get("coordinate"),
        **dec,
    }


def _write_decision(log_fh, seen: Dict[str, dict], rec: dict) -> None:
    log_fh.write(json.dumps(rec, default=str) + "\n")
    log_fh.flush()
    seen[rec["pair_key"]] = rec


def judge_phase(args: argparse.Namespace, files: List[str],
                log_path: str) -> Dict[str, dict]:
    seen = _load_decisions_log(log_path)
    print(f"[judge] resume: {len(seen)} pairs already decided in {log_path}")

    # Only the trajectory column is needed to discover pairs.
    log_fh = open(log_path, "a")
    n_new = 0
    try:
        if args.num_workers <= 1:
            client = _make_client(args.provider)
            for fp in files:
                fname = os.path.basename(fp)
                pf = pq.ParquetFile(fp)
                ri = 0
                stop = False
                file_pbar = tqdm(desc=f"judge {fname}", unit="pair", leave=False)
                for batch in pf.iter_batches(batch_size=32, columns=["trajectory"]):
                    if stop:
                        break
                    df_batch = batch.to_pandas()
                    for _, row in df_batch.iterrows():
                        if args.rows_per_file is not None and ri >= args.rows_per_file:
                            stop = True
                            break
                        traj = _parse_traj(row["trajectory"])
                        for _, ck, tk, click, typ in _find_pairs(traj):
                            key = _pair_key(fname, ri, ck, tk)
                            if key in seen:
                                continue
                            dec = _judge(client, args.model, args.provider, click, typ)
                            rec = {
                                "pair_key": key,
                                "file": fname,
                                "traj_idx": int(ri),
                                "click_step_key": str(ck),
                                "type_step_key": str(tk),
                                "click_bbox": click["args"].get("bbox")
                                           or click["args"].get("coordinate"),
                                **dec,
                            }
                            _write_decision(log_fh, seen, rec)
                            n_new += 1
                            file_pbar.update(1)
                        ri += 1
                file_pbar.close()
            print(f"[judge] done: +{n_new} new decisions; total {len(seen)} in log")
            return seen

        max_pending = max(args.num_workers * 4, 1)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_workers
        ) as executor:
            pending: Dict[concurrent.futures.Future, tqdm] = {}

            def drain_one() -> None:
                nonlocal n_new
                done, _ = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for fut in done:
                    pbar = pending.pop(fut)
                    rec = fut.result()
                    _write_decision(log_fh, seen, rec)
                    n_new += 1
                    pbar.update(1)

            for fp in files:
                fname = os.path.basename(fp)
                pf = pq.ParquetFile(fp)
                ri = 0
                stop = False
                file_pbar = tqdm(desc=f"judge {fname}", unit="pair", leave=False)
                try:
                    for batch in pf.iter_batches(batch_size=32, columns=["trajectory"]):
                        if stop:
                            break
                        df_batch = batch.to_pandas()
                        for _, row in df_batch.iterrows():
                            if args.rows_per_file is not None and ri >= args.rows_per_file:
                                stop = True
                                break
                            traj = _parse_traj(row["trajectory"])
                            for _, ck, tk, click, typ in _find_pairs(traj):
                                key = _pair_key(fname, ri, ck, tk)
                                if key in seen:
                                    continue
                                while len(pending) >= max_pending:
                                    drain_one()
                                fut = executor.submit(
                                    _judge_pair, args, fname, ri, ck, tk, click, typ
                                )
                                pending[fut] = file_pbar
                            ri += 1
                    while pending:
                        drain_one()
                finally:
                    file_pbar.close()

    finally:
        log_fh.close()

    print(f"[judge] done: +{n_new} new decisions; total {len(seen)} in log")
    return seen


def _bbox_center(bbox: Any) -> Optional[List[float]]:
    if not bbox or len(bbox) != 4:
        return None
    x, y, w, h = [float(v) for v in bbox]
    return [x + w / 2, y + h / 2]


def _images_to_list(images: Any) -> List[Any]:
    if images is None:
        return []
    if isinstance(images, float) and pd.isna(images):
        return []
    return list(images)


def _apply_to_traj(traj: Dict[str, Any], images: List[Any],
                   per_row: List[dict]) -> Optional[Tuple[Dict[str, Any], List[Any]]]:
    """Apply merge decisions to a single trajectory/images pair. Returns
    (new_traj, new_images) if anything changed, else None.
    """
    step_keys = sorted(traj.keys(),
                       key=lambda k: int(k) if str(k).isdigit() else 0)

    drop_keys: set = set()
    updates: Dict[str, dict] = {}
    for dec in per_row:
        if not dec.get("merge"):
            continue
        ck = dec["click_step_key"]
        tk = dec["type_step_key"]
        if ck not in traj or tk not in traj:
            continue
        drop_keys.add(ck)
        upd: Dict[str, Any] = {
            "thought": dec.get("merged_thought") or "",
        }
        bbox = dec.get("click_bbox")
        if bbox:
            upd["bbox"] = bbox
            center = _bbox_center(bbox)
            if center is not None:
                upd["coordinate"] = center
        updates[tk] = upd

    if not drop_keys and not updates:
        return None

    new_traj: Dict[str, Any] = {}
    new_images: List[Any] = []
    new_idx = 1
    for pos, k in enumerate(step_keys):
        if k in drop_keys:
            continue
        step = copy.deepcopy(traj[k])
        if k in updates:
            ao = step.setdefault("action", {}).setdefault("action_output", {})
            upd = updates[k]
            ao["action_name"] = "type"
            if upd.get("thought"):
                ao["thought"] = upd["thought"]
            inner = ao.setdefault("action", {})
            if "bbox" in upd:
                inner["bbox"] = upd["bbox"]
            if "coordinate" in upd:
                inner["coordinate"] = upd["coordinate"]
            action_str = step.get("action", {}).get("action_str")
            if isinstance(action_str, str) and action_str.startswith("keyboard_type("):
                step["action"]["action_str"] = "type(" + action_str[len("keyboard_type("):]
        new_traj[str(new_idx)] = step
        new_idx += 1
        if pos < len(images):
            new_images.append(images[pos])
    return new_traj, new_images


def apply_phase(args: argparse.Namespace, files: List[str],
                decisions: Dict[str, dict]) -> None:
    by_row: Dict[Tuple[str, int], List[dict]] = {}
    files_with_merges: set = set()
    for dec in decisions.values():
        by_row.setdefault((dec["file"], int(dec["traj_idx"])), []).append(dec)
        if dec.get("merge"):
            files_with_merges.add(dec["file"])
    print(f"[apply] decisions cover {len(by_row)} rows; "
          f"{len(files_with_merges)} files contain at least one merge=true")

    os.makedirs(args.out_dir, exist_ok=True)
    n_modified = 0
    for fp in files:
        fname = os.path.basename(fp)
        out_path = os.path.join(args.out_dir, fname)
        if fname not in files_with_merges:
            # No mutations needed; copy the original through.
            if os.path.abspath(fp) != os.path.abspath(out_path):
                shutil.copyfile(fp, out_path)
            continue

        pf = pq.ParquetFile(fp)
        schema = pf.schema_arrow
        writer = pq.ParquetWriter(out_path, schema)
        ri = 0
        try:
            for batch in tqdm(pf.iter_batches(batch_size=16),
                              desc=f"apply {fname}", leave=False,
                              total=pf.num_row_groups):
                df_batch = batch.to_pandas()
                out_rows: List[Dict[str, Any]] = []
                for _, row in df_batch.iterrows():
                    row_dict = row.to_dict()
                    per_row = by_row.get((fname, ri), [])
                    if per_row:
                        traj = _parse_traj(row_dict["trajectory"])
                        images = _images_to_list(row_dict.get("images"))
                        result = _apply_to_traj(traj, images, per_row)
                        if result is not None:
                            new_traj, new_images = result
                            row_dict["trajectory"] = json.dumps(new_traj)
                            row_dict["images"] = new_images
                            n_modified += 1
                    out_rows.append(row_dict)
                    ri += 1
                out_df = pd.DataFrame(out_rows)
                table = pa.Table.from_pandas(out_df, schema=schema,
                                             preserve_index=False)
                writer.write_table(table)
        finally:
            writer.close()
    print(f"[apply] modified {n_modified} rows; output in {args.out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        default="/gpfs/scrubbed/reza/MolmoWeb-SyntheticTrajs/data_filtered",
    )
    p.add_argument(
        "--out_dir",
        default="/gpfs/scrubbed/reza/MolmoWeb-SyntheticTrajs/data_filtered_typemerged",
    )
    p.add_argument(
        "--decisions_log",
        default="/gpfs/scrubbed/reza/fara/viz/click_type_merge/decisions_full.jsonl",
    )
    p.add_argument("--provider", choices=["anthropic", "google"], default="google")
    p.add_argument("--model", default=None,
                   help="Defaults: claude-haiku-4-5 (anthropic), gemini-2.5-flash (google).")
    p.add_argument("--max_files", type=int, default=None,
                   help="Only process the first N parquet files (debug).")
    p.add_argument("--rows_per_file", type=int, default=None,
                   help="Judge phase only: cap rows judged per file (debug).")
    p.add_argument("--num_workers", type=int, default=1,
                   help="Concurrent LLM judge calls. Use 1 for sequential judging.")
    p.add_argument("--skip_judge", action="store_true",
                   help="Skip Phase 1; reuse decisions already in --decisions_log.")
    p.add_argument("--skip_apply", action="store_true",
                   help="Skip Phase 2; only build the decisions log.")
    args = p.parse_args()

    if args.model is None:
        args.model = {"anthropic": "claude-haiku-4-5",
                      "google":    "gemini-2.5-flash"}[args.provider]

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.parquet")))
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        sys.exit(f"no parquet files under {args.data_dir}")
    print(f"[main] {len(files)} parquet file(s); provider={args.provider}; "
          f"model={args.model}")

    decisions: Dict[str, dict] = {}
    if not args.skip_judge:
        os.makedirs(os.path.dirname(args.decisions_log) or ".", exist_ok=True)
        decisions = judge_phase(args, files, args.decisions_log)
    else:
        decisions = _load_decisions_log(args.decisions_log)
        print(f"[judge] skipped; loaded {len(decisions)} existing decisions")

    if args.skip_apply:
        print("[apply] skipped")
        return

    apply_phase(args, files, decisions)


if __name__ == "__main__":
    main()
