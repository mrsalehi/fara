"""Render a webvoyager (or any websurfer-style) eval run as browsable HTML.

Reads a run dir containing per-task subdirectories. Each task dir is expected
to have:
  - web_surfer.log     : JSONL of events; the `arguments` dict on each event
                         carries `thoughts` plus the action name/args
  - screenshot{i}.png  : 0-indexed screenshot per step
  - *_final_answer.json: final answer + token usage (optional for viz)

Renders the same layout as `visualize_trajectories.py` so the training-data
viz and the eval-run viz can be compared side by side.

Usage:
    python scripts/visualize_run.py \\
        --run_dir /gpfs/projects/raivn/reza/fara/results/.../traj \\
        --output_dir /gpfs/scrubbed/reza/fara/viz_run \\
        --max_tasks 50
"""
import argparse
import base64
import io
import json
import re
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


USER_MESSAGE = "Here is the next screenshot. Think about what to do next."

PAGE_CSS = """
body { font-family: -apple-system, system-ui, "Segoe UI", sans-serif; background: #f5f5f5;
       margin: 0; padding: 2rem; }
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.3rem; margin: 0 0 0.4rem 0; word-break: break-all; }
.meta { color: #666; font-size: 0.85rem; margin-bottom: 1rem; }
.instruction { background: #fff8e1; padding: 0.8rem 1rem; border-left: 4px solid #ffb300;
               border-radius: 4px; margin-bottom: 1.5rem; }
.final { background: #e8f5e9; padding: 0.8rem 1rem; border-left: 4px solid #43a047;
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


def _img_data_uri(path: Path, max_dim: int) -> str:
    img = Image.open(path).convert("RGB")
    if max_dim and max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _list_screenshots(task_dir: Path) -> List[Path]:
    pat = re.compile(r"^screenshot(\d+)\.png$")
    files: List[Tuple[int, Path]] = []
    for p in task_dir.iterdir():
        m = pat.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort()
    return [p for _, p in files]


def _load_events(task_dir: Path) -> List[Dict[str, Any]]:
    log = task_dir / "web_surfer.log"
    if not log.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in log.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # Keep only WebSurfer events with an action (mirrors the gpt_solver path
    # in webeval/trajectory.py — non-WebSurfer events are summarizer/diag).
    return [
        e for e in out
        if e.get("action") is not None and (
            e.get("source") in (None, "WebSurfer", "fara")
            or "WebSurfer" in str(e.get("source", ""))
        )
    ]


def _load_final_answer(task_dir: Path) -> Optional[Dict[str, Any]]:
    matches = list(task_dir.glob("*_final_answer.json")) or list(task_dir.glob("*_answer.json"))
    if not matches:
        return None
    try:
        return json.loads(matches[0].read_text())
    except json.JSONDecodeError:
        return None


def _load_question(task_dir: Path) -> str:
    """Best-effort task-instruction lookup. The harness sometimes drops a
    metadata.json next to the screenshots; otherwise fall back to scraping the
    first relevant line out of core.log.
    """
    meta = task_dir / "metadata.json"
    if meta.exists():
        try:
            data = json.loads(meta.read_text())
            for key in ("question", "instruction", "task", "ques"):
                v = data.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        except json.JSONDecodeError:
            pass
    core = task_dir / "core.log"
    if core.exists():
        for line in core.read_text().splitlines():
            m = re.search(r"question[_ ]?text[\"']?\s*[:=]\s*[\"'](.+?)[\"']", line, re.I)
            if m:
                return m.group(1).strip()
    return ""


def _action_summary(event: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Return (thought, pretty-printable action dict)."""
    args = event.get("arguments") or {}
    thought = ""
    if isinstance(args, dict):
        thought = str(args.get("thoughts") or args.get("thought") or "").strip()
        action = {k: v for k, v in args.items() if k not in ("thoughts", "thought")}
        # Always include the action name at the top.
        if event.get("action") and "action" not in action:
            action = {"action": event["action"], **action}
    else:
        action = {"action": event.get("action", ""), "arguments": args}
    return thought, action


def _step_url(event: Dict[str, Any]) -> str:
    args = event.get("arguments")
    if isinstance(args, dict):
        for k in ("url", "current_url", "page_url"):
            v = args.get(k)
            if isinstance(v, str):
                return v
    for k in ("url", "current_url", "page_url"):
        v = event.get(k)
        if isinstance(v, str):
            return v
    return ""


def _render_task(task_dir: Path, max_image_dim: int) -> Optional[str]:
    screenshots = _list_screenshots(task_dir)
    events = _load_events(task_dir)
    if not screenshots and not events:
        return None
    answer = _load_final_answer(task_dir)
    question = _load_question(task_dir)
    final = answer.get("final_answer") if isinstance(answer, dict) else None

    n = min(len(screenshots), len(events)) if events else len(screenshots)
    out: List[str] = []
    out.append(
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>{escape(task_dir.name)}</title>'
        f'<style>{PAGE_CSS}</style></head><body><div class="wrap">'
    )
    out.append(f'<h1>{escape(task_dir.name)}</h1>')
    out.append(
        f'<div class="meta">{len(screenshots)} screenshot(s), '
        f'{len(events)} action(s) · '
        f'<a href="index.html">← back to index</a></div>'
    )
    if question:
        out.append(f'<div class="instruction"><strong>Task:</strong> '
                   f'{escape(question)}</div>')
    if final and final != "<no_answer>":
        out.append(f'<div class="final"><strong>Final answer:</strong> '
                   f'{escape(str(final))}</div>')

    # Pair screenshot[i] with event[i]. screenshot0 is the page state BEFORE
    # action 0 (matching the training convention: user(image) → assistant(action)).
    for i in range(n):
        img_uri = _img_data_uri(screenshots[i], max_image_dim)
        ev = events[i] if i < len(events) else {}
        url = _step_url(ev)
        thought, action = _action_summary(ev)

        # User turn — first turn carries the question; later turns use the
        # production "next screenshot" prompt.
        if i == 0:
            user_text = question or "(no question text recovered)"
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
        if thought:
            out.append(f'<div class="thought">{escape(thought)}</div>')
        action_str = json.dumps(action, indent=2, ensure_ascii=False, default=str)
        out.append(f'<pre>{escape(action_str)}</pre>')
        out.append('</div>')

    # Trailing screenshots (e.g., final state captured after the last action).
    for i in range(n, len(screenshots)):
        img_uri = _img_data_uri(screenshots[i], max_image_dim)
        out.append('<div class="turn user">')
        out.append(f'<div class="role">User · step {i + 1} (no matching event)</div>')
        out.append(f'<img src="{img_uri}" alt="screenshot {i + 1}">')
        out.append('</div>')

    out.append('</div></body></html>')
    return "".join(out)


def _safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", str(name))[:120]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True,
                   help="Path containing per-task subdirectories (e.g. .../traj).")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_tasks", type=int, default=None)
    p.add_argument("--max_image_dim", type=int, default=1280,
                   help="Resize each screenshot so its longest side is at "
                        "most this many pixels. 0 = keep original.")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_dirs = sorted(d for d in run_dir.iterdir() if d.is_dir())
    if not task_dirs:
        raise SystemExit(f"no task subdirectories under {run_dir}")
    if args.max_tasks is not None:
        task_dirs = task_dirs[: args.max_tasks]

    idx_parts: List[str] = [
        '<!doctype html><html><head><meta charset="utf-8">'
        '<title>Run Index</title>'
        f'<style>{PAGE_CSS}</style></head><body><div class="wrap">'
        f'<h1>{escape(run_dir.name)}</h1><ol>'
    ]

    written = 0
    for i, td in enumerate(task_dirs):
        html = _render_task(td, args.max_image_dim)
        if html is None:
            print(f"[skip] {td.name}: no screenshots or events")
            continue
        fname = _safe_filename(td.name) + ".html"
        (out_dir / fname).write_text(html)
        question = _load_question(td)
        question_short = (question[:140] + "…") if len(question) > 140 else question
        idx_parts.append(
            f'<li><a href="{escape(fname)}">{escape(td.name)}</a>'
            f' <span style="color:#666"> — {escape(question_short)}</span></li>'
        )
        written += 1
        print(f"[{i + 1}/{len(task_dirs)}] wrote {fname}")

    idx_parts.append('</ol></div></body></html>')
    (out_dir / "index.html").write_text("\n".join(idx_parts))
    print(f"\n{written} task(s) rendered. open {out_dir}/index.html")


if __name__ == "__main__":
    main()
