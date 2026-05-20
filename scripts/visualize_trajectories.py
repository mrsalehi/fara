"""Render FARA SFT trajectories as browsable HTML.

For each trajectory row, emits a self-contained HTML page showing:
  - the task instruction
  - alternating user (screenshot + prompt text) / assistant (thought + action)
    turns, in the same shape the training pipeline feeds the model
  - the URL pulled from each step (when available)

Images are embedded as base64 PNG data URIs so the output dir is portable
(no server, no broken links). An index.html lists every row by sample_id.

Usage:
    python scripts/visualize_trajectories.py \\
        --data_path /gpfs/scrubbed/reza/MolmoWeb-SyntheticTrajs/data_filtered/ \\
        --output_dir /gpfs/scrubbed/reza/fara/viz \\
        --max_rows 50 \\
        --shuffle_seed 0
"""
import argparse
import base64
import bisect
from collections import defaultdict
import io
import json
import random
import re
import sys
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
from datasets import load_dataset
from PIL import Image

# Import the training pipeline's MolmoWeb→Fara action translator so the viz
# shows exactly what the model is taught to emit. train_fara.py guards its
# main() under __name__ == "__main__", so this import is side-effect-free.
sys.path.insert(0, str(Path(__file__).parent))
from train_fara import (  # noqa: E402
    _domain_allowed,
    _extract_row_domains,
    _format_assistant_message,
)


USER_MESSAGE = "Here is the next screenshot. Think about what to do next."


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return list(value)


def _to_pil(img_entry: Any) -> Image.Image:
    if isinstance(img_entry, Image.Image):
        return img_entry.convert("RGB")
    if isinstance(img_entry, dict) and img_entry.get("bytes") is not None:
        return Image.open(io.BytesIO(img_entry["bytes"])).convert("RGB")
    if isinstance(img_entry, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(img_entry))).convert("RGB")
    if isinstance(img_entry, np.ndarray):
        arr = img_entry if img_entry.dtype == np.uint8 else img_entry.astype(np.uint8)
        return Image.fromarray(arr)
    raise TypeError(f"unsupported image type: {type(img_entry)}")


def _img_data_uri(img_entry: Any, max_dim: int) -> str:
    img = _to_pil(img_entry)
    if max_dim and max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _extract_instruction(raw: Any) -> str:
    if isinstance(raw, str):
        try:
            return _extract_instruction(json.loads(raw))
        except json.JSONDecodeError:
            return raw.strip()
    if isinstance(raw, dict):
        for key in ("high_level", "instruction", "low_level", "task", "goal"):
            v = raw.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for v in raw.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def _parse_trajectory(raw: Any) -> List[Any]:
    """Trajectory may be a JSON string, a 1-indexed dict, or a list — normalize
    to an in-order list of step dicts.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if isinstance(raw, dict):
        return [raw[k] for k in sorted(raw.keys(), key=lambda k: int(k))]
    if isinstance(raw, list):
        return raw
    return []


def _step_field(step: Any, *paths) -> Any:
    """Walk nested dict paths, returning the first non-empty match. Each path
    is a tuple of keys, e.g. ('action', 'action_output', 'action_name').
    """
    if not isinstance(step, dict):
        return None
    for path in paths:
        cur: Any = step
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok and cur not in (None, "", {}, []):
            return cur
    return None


def _step_url(step: Any) -> str:
    v = _step_field(
        step,
        ("url",), ("current_url",), ("page_url",),
        ("observation", "url"), ("observation", "current_url"), ("observation", "page_url"),
    )
    return v if isinstance(v, str) else ""


PAGE_CSS = """
body { font-family: -apple-system, system-ui, "Segoe UI", sans-serif; background: #f5f5f5;
       margin: 0; padding: 2rem; }
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.4rem; margin: 0 0 1rem 0; }
h2 { font-size: 1.15rem; margin: 0; word-break: break-all; }
.meta { color: #666; font-size: 0.85rem; margin-bottom: 1rem; }
.toc { background: #fff; padding: 1rem 1.2rem; border-radius: 6px; margin-bottom: 2rem;
       border: 1px solid #e0e0e0; }
.toc ol { margin: 0.4rem 0 0 1.5rem; padding: 0; }
.toc li { margin: 0.2rem 0; }
details.traj { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;
               padding: 0.8rem 1.2rem; margin-bottom: 1rem; }
details.traj > summary { cursor: pointer; padding: 0.2rem 0; }
details.traj[open] > summary { margin-bottom: 1rem; border-bottom: 1px solid #eee;
                                padding-bottom: 0.6rem; }
.instruction { background: #fff8e1; padding: 0.8rem 1rem; border-left: 4px solid #ffb300;
               border-radius: 4px; margin-bottom: 1.5rem; }
.turn { margin-bottom: 1.2rem; padding: 0.8rem 1rem; border-radius: 6px; }
.user { background: #e3f2fd; border-left: 4px solid #1976d2; }
.assistant { background: #f1f8e9; border-left: 4px solid #689f38; }
.role { font-weight: 600; font-size: 0.8rem; text-transform: uppercase;
        color: #555; margin-bottom: 0.5rem; letter-spacing: 0.05em; }
.text { white-space: pre-wrap; word-break: break-word; }
img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px;
      margin-top: 0.5rem; display: block; }
pre { background: #fafafa; border: 1px solid #ddd; padding: 0.6rem;
      overflow-x: auto; font-size: 0.82rem; margin: 0.4rem 0 0 0;
      white-space: pre-wrap; word-break: break-word; }
.url { color: #555; font-size: 0.8rem; font-family: ui-monospace, monospace;
       margin-bottom: 0.3rem; word-break: break-all; }
.thought { color: #2e3d2a; font-style: italic; margin-bottom: 0.4rem;
           white-space: pre-wrap; }
a { color: #1976d2; text-decoration: none; }
a:hover { text-decoration: underline; }
"""


def _render_row_section(row: Dict[str, Any], row_idx: int,
                        max_image_dim: int, anchor: str,
                        raw: bool = False) -> Tuple[str, str, str]:
    """Render one trajectory as a <details> section. Returns
    (html_section, sample_id_str, instruction_short)."""
    sample_id = str(row.get("sample_id", f"row_{row_idx}"))
    instruction = _extract_instruction(row.get("instruction", ""))
    images = _as_list(row.get("images"))
    steps = _parse_trajectory(row.get("trajectory") or [])
    n = min(len(images), len(steps))

    out: List[str] = []
    out.append(f'<details class="traj" id="{escape(anchor)}">')
    summary_text = f"#{row_idx + 1} · {sample_id} ({n} steps)"
    out.append(f'<summary><h2 style="display:inline">{escape(summary_text)}</h2></summary>')
    if instruction:
        out.append(f'<div class="instruction"><strong>Task:</strong> '
                   f'{escape(instruction)}</div>')

    for i in range(n):
        img_uri = _img_data_uri(images[i], max_image_dim)
        url = _step_url(steps[i])

        # User turn — first turn carries the task instruction; later turns
        # use the production "next screenshot" prompt.
        if i == 0:
            user_text = instruction or "(no instruction)"
            url_line = ""
        else:
            user_text = USER_MESSAGE
            url_line = (f'<div class="url">Current URL: {escape(url)}</div>'
                        if url else "")
        out.append('<div class="turn user">')
        out.append(f'<div class="role">User · step {i + 1}</div>')
        out.append(url_line)
        out.append(f'<div class="text">{escape(user_text)}</div>')
        out.append(f'<img src="{img_uri}" alt="screenshot {i + 1}">')
        out.append('</div>')

        out.append('<div class="turn assistant">')
        out.append(f'<div class="role">Assistant · step {i + 1}</div>')
        raw_str = json.dumps(steps[i], indent=2, ensure_ascii=False, default=str)
        if raw:
            # Show the original MolmoWeb step dict as-is; no translation, no
            # filter — useful for inspecting the source corpus before any
            # preprocessing pass.
            out.append(f'<pre>{escape(raw_str)}</pre>')
        else:
            # Show the exact MolmoWeb→Fara translation the training pipeline
            # emits. None means the step has no fara mapping — flag visibly
            # so the viewer can see why the trajectory would be dropped at
            # training time.
            translated = _format_assistant_message(steps[i])
            if translated is None:
                out.append('<div class="thought" style="color:#b71c1c">'
                           '⚠ no fara mapping for this action — this trajectory '
                           'would be dropped at training time.</div>')
            else:
                out.append(f'<pre>{escape(translated)}</pre>')
            out.append(
                '<details style="margin-top:0.5rem"><summary style="font-size:0.8rem;color:#666">'
                'raw step</summary>'
                f'<pre style="font-size:0.78rem">{escape(raw_str)}</pre></details>'
            )
        out.append('</div>')

    out.append('</details>')
    return "".join(out), sample_id, (instruction[:140] + "…") if len(instruction) > 140 else instruction


def _render_messages_row_section(row: Dict[str, Any], row_idx: int,
                                 max_image_dim: int, anchor: str) -> Tuple[str, str, str]:
    """Render a row in the post-`row_to_messages` format ({messages, images})
    as a <details> section. Walks `messages` in order, pulling images off the
    parallel `images` array as image-typed content blocks are encountered.

    NOTE: this helper exists so train_fara.py can call `render_messages_dataset`
    from inside `build_dataset` (right after the unmapped-action filter) to dump
    a sanity-check HTML of the actual rows the trainer will consume.
    """
    messages = list(row.get("messages") or [])
    images = _as_list(row.get("images"))
    sample_id = str(row.get("sample_id", f"row_{row_idx}"))

    # Best-effort instruction recovery: it's the text of the first user message.
    instruction = ""
    for m in messages:
        if m.get("role") == "user":
            for c in m.get("content", []):
                if isinstance(c, dict) and c.get("type") == "text" and c.get("text"):
                    instruction = c["text"].strip()
                    break
            break

    n_user = sum(1 for m in messages if m.get("role") == "user")
    out: List[str] = []
    out.append(f'<details class="traj" id="{escape(anchor)}">')
    summary_text = f"#{row_idx + 1} · {sample_id} ({n_user} steps)"
    out.append(f'<summary><h2 style="display:inline">{escape(summary_text)}</h2></summary>')
    if instruction:
        out.append(f'<div class="instruction"><strong>Task:</strong> '
                   f'{escape(instruction)}</div>')

    img_idx = 0
    user_step = 0
    asst_step = 0
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", [])
        if role == "system":
            text = "".join(
                c.get("text", "") for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
            out.append(
                '<details style="margin-bottom:1rem"><summary style="font-size:0.85rem;'
                'color:#666">system prompt</summary>'
                f'<pre style="font-size:0.78rem">{escape(text)}</pre></details>'
            )
            continue
        if role == "user":
            user_step += 1
            text_parts = [c.get("text", "") for c in content
                          if isinstance(c, dict) and c.get("type") == "text"]
            user_text = "\n".join(t for t in text_parts if t)
            n_imgs_here = sum(1 for c in content
                              if isinstance(c, dict) and c.get("type") == "image")
            out.append('<div class="turn user">')
            out.append(f'<div class="role">User · turn {user_step}</div>')
            out.append(f'<div class="text">{escape(user_text)}</div>')
            for _ in range(n_imgs_here):
                if img_idx < len(images):
                    img_uri = _img_data_uri(images[img_idx], max_image_dim)
                    out.append(f'<img src="{img_uri}" alt="screenshot {img_idx + 1}">')
                    img_idx += 1
            out.append('</div>')
            continue
        if role == "assistant":
            asst_step += 1
            text = "".join(c.get("text", "") for c in content
                           if isinstance(c, dict) and c.get("type") == "text")
            out.append('<div class="turn assistant">')
            out.append(f'<div class="role">Assistant · turn {asst_step}</div>')
            out.append(f'<pre>{escape(text)}</pre>')
            out.append('</div>')
            continue

    out.append('</details>')
    instr_short = (instruction[:140] + "…") if len(instruction) > 140 else instruction
    return "".join(out), sample_id, instr_short


def render_messages_dataset(rows, output_path, max_image_dim: int = 1280) -> None:
    """Render an iterable of post-`row_to_messages` rows to a single self-contained HTML file.

    Intended caller: `scripts/train_fara.py::build_dataset`, invoked right after
    the unmapped-action filter, so the dumped HTML reflects exactly what the
    trainer will iterate over.
    """
    sections: List[str] = []
    toc_entries: List[str] = []
    for i, row in enumerate(rows):
        anchor = f"traj-{i}"
        section_html, sample_id, instr_short = _render_messages_row_section(
            row, i, max_image_dim, anchor,
        )
        sections.append(section_html)
        toc_entries.append(
            f'<li><a href="#{escape(anchor)}">{escape(sample_id)}</a>'
            f' <span style="color:#666"> — {escape(instr_short)}</span></li>'
        )
    parts: List[str] = [
        '<!doctype html><html><head><meta charset="utf-8">',
        '<title>Trajectories (post-build)</title>',
        f'<style>{PAGE_CSS}</style></head><body><div class="wrap">',
        f'<h1>Trajectories — post-build_dataset ({len(sections)})</h1>',
        '<div class="toc"><strong>Index</strong><ol>',
        *toc_entries,
        '</ol></div>',
        *sections,
        '</div></body></html>',
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts))


def _safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", str(name))[:120]


def _row_allowed(row: Dict[str, Any], allowed: List[str], mode: str) -> bool:
    if not allowed:
        return True
    domains = _extract_row_domains(row.get("trajectory"))
    if not domains:
        return False
    if mode == "strict":
        return all(_domain_allowed(d, allowed) for d in domains)
    return any(_domain_allowed(d, allowed) for d in domains)


def _normalize_allowed_domains(raw: str) -> List[str]:
    if not raw:
        return []
    allowed = [d.strip().lower() for d in raw.split(",") if d.strip()]
    return [d[4:] if d.startswith("www.") else d for d in allowed]


def _file_row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def _sample_parquet_rows(
    files: List[Path],
    max_rows: int,
    seed: int,
    allowed: List[str],
    domain_filter_mode: str,
    oversample_factor: int,
) -> List[Dict[str, Any]]:
    """Randomly sample row positions and load only those rows from parquet.

    This is intended for visualization, not exact train/eval sampling. Without
    a domain filter, sampled rows are uniform over all rows. With a domain
    filter, we oversample first and keep matching rows until max_rows.
    """
    rng = random.Random(seed)
    row_counts = [_file_row_count(f) for f in files]
    total_rows = sum(row_counts)
    if total_rows == 0:
        return []

    target = max_rows if max_rows is not None else total_rows
    sample_size = target
    if allowed:
        sample_size = min(total_rows, max(target * max(oversample_factor, 1), target))
    else:
        sample_size = min(total_rows, target)

    global_indices = rng.sample(range(total_rows), sample_size)
    cumulative = [0]
    for n in row_counts:
        cumulative.append(cumulative[-1] + n)

    by_file: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for order, global_idx in enumerate(global_indices):
        file_idx = bisect.bisect_right(cumulative, global_idx) - 1
        local_idx = global_idx - cumulative[file_idx]
        by_file[file_idx].append((order, local_idx))

    sampled: List[Tuple[int, Dict[str, Any]]] = []
    for file_idx, order_and_rows in by_file.items():
        fp = files[file_idx]
        pf = pq.ParquetFile(fp)
        rg_starts = [0]
        for rg_idx in range(pf.num_row_groups):
            rg_starts.append(rg_starts[-1] + pf.metadata.row_group(rg_idx).num_rows)

        by_rg: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for order, local_idx in order_and_rows:
            rg_idx = bisect.bisect_right(rg_starts, local_idx) - 1
            offset = local_idx - rg_starts[rg_idx]
            by_rg[rg_idx].append((order, offset))

        for rg_idx, order_and_offsets in by_rg.items():
            table = pf.read_row_group(rg_idx)
            rows = table.to_pylist()
            for order, offset in order_and_offsets:
                row = dict(rows[offset])
                row.setdefault("sample_id", f"{fp.name}:{rg_starts[rg_idx] + offset}")
                row["_source_file"] = fp.name
                row["_source_row"] = rg_starts[rg_idx] + offset
                if _row_allowed(row, allowed, domain_filter_mode):
                    sampled.append((order, row))

    sampled.sort(key=lambda x: x[0])
    rows = [row for _, row in sampled]
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True,
                   help="Parquet file or directory of parquets.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_rows", type=int, default=50)
    p.add_argument("--shuffle_seed", type=int, default=0,
                   help="Seed for shuffling before slicing. Set to a negative "
                        "number to skip the shuffle (parquet file order). "
                        "Default 0 keeps the sample reproducible.")
    p.add_argument("--sample_mode", choices=("hf", "parquet_random"), default="hf",
                   help="'hf' uses datasets.load_dataset then shuffles/selects. "
                        "'parquet_random' samples row positions directly from "
                        "parquet files and loads only those rows.")
    p.add_argument("--parquet_random_oversample_factor", type=int, default=10,
                   help="When --sample_mode parquet_random and domain filtering "
                        "is enabled, sample max_rows*N candidate rows before "
                        "filtering.")
    p.add_argument("--max_image_dim", type=int, default=1280,
                   help="Resize each screenshot so its longest side is at "
                        "most this many pixels. 0 = keep original.")
    p.add_argument("--allowed_domains", type=str, default=None,
                   help="Comma-separated allowlist of domains "
                        "(e.g. 'arxiv.org,allrecipes.com'). When set, rows "
                        "whose trajectory URLs do not match are dropped — "
                        "matches train_fara's --allowed_domains semantics.")
    p.add_argument("--domain_filter_mode", choices=("any", "strict"), default="any",
                   help="'any': keep row if any step URL matches the allowlist. "
                        "'strict': keep row only if ALL step URLs match.")
    p.add_argument("--raw", action="store_true",
                   help="Show the raw MolmoWeb step dict instead of the "
                        "MolmoWeb→Fara translation. Pair with --data_path "
                        "pointing at the un-preprocessed source dir to view "
                        "the corpus before any filtering.")
    args = p.parse_args()

    in_path = Path(args.data_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [in_path] if in_path.is_file() else sorted(in_path.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {in_path}")

    allowed = _normalize_allowed_domains(args.allowed_domains)
    if args.sample_mode == "parquet_random":
        if args.shuffle_seed is None or args.shuffle_seed < 0:
            raise SystemExit("--sample_mode parquet_random requires a non-negative --shuffle_seed")
        ds = _sample_parquet_rows(
            files=files,
            max_rows=args.max_rows,
            seed=args.shuffle_seed,
            allowed=allowed,
            domain_filter_mode=args.domain_filter_mode,
            oversample_factor=args.parquet_random_oversample_factor,
        )
        print(f"parquet_random sample: {len(ds)} row(s) selected")
        if len(ds) == 0:
            raise SystemExit("random sampling produced no rows.")
    else:
        ds = load_dataset(
            "parquet", data_files=[str(f) for f in files], split="train",
        )

        # Domain allowlist — same logic as train_fara's _keep_row, kept verbatim
        # so the viz mirrors what training would actually consume.
        if allowed:
            def _keep(row):
                return _row_allowed(row, allowed, args.domain_filter_mode)
            before = len(ds)
            ds = ds.filter(_keep, num_proc=4)
            print(f"domain filter ({args.domain_filter_mode}, {allowed}): "
                  f"{len(ds)}/{before} rows kept")
            if len(ds) == 0:
                raise SystemExit("domain filter removed all rows.")

        if args.shuffle_seed is not None and args.shuffle_seed >= 0:
            ds = ds.shuffle(seed=args.shuffle_seed)
        if args.max_rows is not None:
            ds = ds.select(range(min(args.max_rows, len(ds))))

    # Render each trajectory as a <details> block, then assemble one HTML
    # document: TOC at the top, all sections below. Single file = single URL
    # to open in the browser, no Live Server hop required.
    sections: List[str] = []
    toc_entries: List[str] = []
    for i, row in enumerate(ds):
        anchor = f"traj-{i}"
        section_html, sample_id, instr_short = _render_row_section(
            row, i, args.max_image_dim, anchor, raw=args.raw,
        )
        sections.append(section_html)
        toc_entries.append(
            f'<li><a href="#{escape(anchor)}">{escape(sample_id)}</a>'
            f' <span style="color:#666"> — {escape(instr_short)}</span></li>'
        )
        print(f"[{i + 1}/{len(ds)}] rendered {sample_id}")

    parts: List[str] = [
        '<!doctype html><html><head><meta charset="utf-8">',
        '<title>Trajectories</title>',
        f'<style>{PAGE_CSS}</style></head><body><div class="wrap">',
        f'<h1>Trajectories ({len(sections)})</h1>',
        '<div class="toc"><strong>Index</strong><ol>',
        *toc_entries,
        '</ol></div>',
        *sections,
        '</div></body></html>',
    ]
    out_path = out_dir / "index.html"
    out_path.write_text("\n".join(parts))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
