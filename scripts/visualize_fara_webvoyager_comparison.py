#!/usr/bin/env python
"""Build an HTML dashboard comparing FARA WebVoyager runs and training data.

This is a light wrapper around:
  - scripts/visualize_run.py for WebSurfer/WebVoyager eval trajectories
  - scripts/visualize_trajectories.py for MolmoWeb synthetic training rows

It writes one dashboard with links to both rendered views.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from html import escape
from pathlib import Path
from typing import Iterable, Optional


PAGE_CSS = """
body { font-family: -apple-system, system-ui, "Segoe UI", sans-serif; background: #f5f5f5;
       margin: 0; padding: 2rem; color: #222; }
.wrap { max-width: 980px; margin: 0 auto; }
h1 { font-size: 1.35rem; margin: 0 0 0.6rem 0; }
.meta { color: #666; font-size: 0.9rem; margin-bottom: 1.4rem; }
.panel { background: #fff; border: 1px solid #ddd; border-radius: 6px;
         padding: 1rem 1.2rem; margin: 1rem 0; }
.panel h2 { font-size: 1.05rem; margin: 0 0 0.5rem 0; }
.path { color: #555; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: 0.82rem; word-break: break-all; margin-top: 0.5rem; }
a { color: #1976d2; text-decoration: none; font-weight: 600; }
a:hover { text-decoration: underline; }
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _has_task_dirs(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and (
            (child / "web_surfer.log").exists()
            or any(child.glob("screenshot*.png"))
        ):
            return True
    return False


def _resolve_traj_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if _has_task_dirs(path):
        return path
    if _has_task_dirs(path / "traj"):
        return path / "traj"

    candidates = [p for p in path.rglob("traj") if _has_task_dirs(p)]
    if not candidates:
        raise SystemExit(
            f"Could not find a WebVoyager traj directory under {path}. "
            "Pass the directory that contains task folders like ArXiv--0."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _run(cmd: Iterable[str], env: Optional[dict] = None) -> None:
    print("[cmd]", " ".join(str(x) for x in cmd))
    subprocess.run(list(cmd), check=True, env=env)


def _write_dashboard(
    out_dir: Path,
    traj_dir: Path,
    training_path: Optional[Path],
    max_tasks: Optional[int],
    training_max_rows: Optional[int],
) -> None:
    training_block = ""
    if training_path is not None:
        training_block = f"""
        <div class="panel">
          <h2>MolmoWeb Synthetic Training Data</h2>
          <a href="molmoweb_training/index.html">Open training-data visualization</a>
          <div class="path">{escape(str(training_path))}</div>
          <div class="meta">Rows rendered: {escape(str(training_max_rows))}</div>
        </div>
        """

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>FARA WebVoyager Comparison</title>
  <style>{PAGE_CSS}</style>
</head>
<body>
  <div class="wrap">
    <h1>FARA WebVoyager Comparison</h1>
    <div class="meta">Rendered outputs are self-contained HTML files with screenshots embedded.</div>
    <div class="panel">
      <h2>FARA WebVoyager Predictions</h2>
      <a href="fara_webvoyager/index.html">Open FARA run visualization</a>
      <div class="path">{escape(str(traj_dir))}</div>
      <div class="meta">Tasks rendered: {escape(str(max_tasks))}</div>
    </div>
    {training_block}
  </div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        required=True,
        help="WebVoyager output root or traj directory containing per-task dirs.",
    )
    parser.add_argument(
        "--training_data_path",
        default="/gpfs/scrubbed/reza/MolmoWeb-SyntheticTrajs/data_filtered",
        help="MolmoWeb parquet file/dir to render for comparison.",
    )
    parser.add_argument(
        "--output_dir",
        default="/gpfs/scrubbed/reza/fara/viz/webvoyager_comparison",
    )
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--training_max_rows", type=int, default=50)
    parser.add_argument("--training_shuffle_seed", type=int, default=0)
    parser.add_argument("--max_image_dim", type=int, default=1280)
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="HF datasets/cache directory. Defaults to <output_dir>/.hf_cache.",
    )
    parser.add_argument(
        "--allowed_domains",
        default=None,
        help="Optional comma-separated domain allowlist for training rows.",
    )
    parser.add_argument(
        "--domain_filter_mode",
        choices=("any", "strict"),
        default="any",
    )
    parser.add_argument(
        "--raw_training",
        action="store_true",
        help="Render raw MolmoWeb steps instead of FARA-translated training messages.",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Only render the FARA WebVoyager run.",
    )
    args = parser.parse_args()

    repo = _repo_root()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        Path(args.cache_dir).expanduser().resolve()
        if args.cache_dir
        else out_dir / ".hf_cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    child_env["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    child_env["HF_HOME"] = str(cache_dir)
    child_env["HF_HUB_CACHE"] = str(cache_dir / "hub")

    traj_dir = _resolve_traj_dir(Path(args.run_dir))
    fara_out = out_dir / "fara_webvoyager"
    train_out = out_dir / "molmoweb_training"

    run_cmd = [
        sys.executable,
        str(repo / "scripts" / "visualize_run.py"),
        "--run_dir",
        str(traj_dir),
        "--output_dir",
        str(fara_out),
        "--max_image_dim",
        str(args.max_image_dim),
    ]
    if args.max_tasks is not None:
        run_cmd.extend(["--max_tasks", str(args.max_tasks)])
    _run(run_cmd, env=child_env)

    training_path = None
    if not args.skip_training:
        training_path = Path(args.training_data_path).expanduser().resolve()
        train_cmd = [
            sys.executable,
            str(repo / "scripts" / "visualize_trajectories.py"),
            "--data_path",
            str(training_path),
            "--output_dir",
            str(train_out),
            "--max_rows",
            str(args.training_max_rows),
            "--shuffle_seed",
            str(args.training_shuffle_seed),
            "--sample_mode",
            "parquet_random",
            "--max_image_dim",
            str(args.max_image_dim),
            "--domain_filter_mode",
            args.domain_filter_mode,
        ]
        if args.allowed_domains:
            train_cmd.extend(["--allowed_domains", args.allowed_domains])
        if args.raw_training:
            train_cmd.append("--raw")
        _run(train_cmd, env=child_env)

    _write_dashboard(
        out_dir=out_dir,
        traj_dir=traj_dir,
        training_path=training_path,
        max_tasks=args.max_tasks,
        training_max_rows=None if args.skip_training else args.training_max_rows,
    )
    print(f"\nOpen {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
