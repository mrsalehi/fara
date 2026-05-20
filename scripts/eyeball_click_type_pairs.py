#!/usr/bin/env python
"""Sample click->type pairs from MolmoWeb trajectories, render an HTML
report for human eyeballing, and (optionally) judge each pair with a
text-only LLM. Output: report.html + decisions.jsonl.
"""
import argparse, base64, glob, html as _html, io, json, os, random, sys
import pandas as pd
from PIL import Image, ImageDraw


def _parse_traj(t):
    return json.loads(t) if isinstance(t, str) else t


def _step_action(step):
    ao = step.get("action", {}).get("action_output", {}) or {}
    return {
        "name": ao.get("action_name", ""),
        "thought": ao.get("thought", ""),
        "args": ao.get("action", {}) or {},
        "action_str": step.get("action", {}).get("action_str", ""),
    }


def _find_pairs(traj):
    keys = sorted(traj.keys(), key=lambda k: int(k) if str(k).isdigit() else 0)
    out = []
    for i in range(len(keys) - 1):
        a = _step_action(traj[keys[i]])
        b = _step_action(traj[keys[i + 1]])
        if a["name"] == "click" and b["name"] == "keyboard_type":
            out.append((i, keys[i], keys[i + 1], a, b))
    return out


def _load_img(entry):
    raw = entry["bytes"] if isinstance(entry, dict) else entry
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _draw_bbox(img, bbox, color="red"):
    if not bbox or len(bbox) != 4:
        return img
    img = img.copy()
    d = ImageDraw.Draw(img)
    x, y, w, h = [float(v) for v in bbox]
    d.rectangle([x, y, x + w, y + h], outline=color, width=4)
    return img


def _img_b64(img, max_w=900):
    if img.width > max_w:
        r = max_w / img.width
        img = img.resize((max_w, int(img.height * r)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise json.JSONDecodeError("No JSON object found", text, 0)


JUDGE_SYS = (
    "You decide whether two consecutive web-agent actions should be merged "
    "into a single canonical type(coordinate, text) action.\n\n"
    "MERGE = true: the click set focus on exactly the input field where the "
    "subsequent type writes text (e.g., click a search box then type a query).\n"
    "MERGE = false: the click opened or activated a different UI element "
    "(button, dropdown, modal trigger, link) and the type then writes into "
    "some other field that appeared.\n\n"
    "When merge=true, also produce `merged_thought`: a SINGLE concise "
    "FARA-style reasoning sentence for the canonical type action. The sentence "
    "should describe entering/inputting/typing the target text into the relevant "
    "field, search bar, textbox, or form control. Do NOT mention clicking, "
    "\"already focused\", or a prior action; the click is represented only by "
    "the type action's coordinate. Good examples: "
    "\"To find the spicy tofu stir-fry recipe, I should enter 'spicy tofu "
    "stir-fry' into the search bar.\" "
    "\"Enter the exact paper title into the search box to query arXiv.\" "
    "Preserve the user's goal/context and the exact text being typed when it is "
    "helpful. When merge=false, set merged_thought to \"\".\n\n"
    "Respond with strict JSON only and nothing else: "
    "{\"merge\": true|false, \"reason\": \"<one short sentence>\", "
    "\"merged_thought\": \"<sentence or empty string>\"}"
)


def _judge(client, model, provider, click, typ):
    user = (
        f"CLICK action_str: {click['action_str']}\n"
        f"CLICK thought: {click['thought']}\n"
        f"CLICK args: {json.dumps(click['args'], default=str)}\n\n"
        f"TYPE action_str: {typ['action_str']}\n"
        f"TYPE thought: {typ['thought']}\n"
        f"TYPE args: {json.dumps(typ['args'], default=str)}"
    )
    
    if provider == "anthropic":
        resp = client.messages.create(
            model=model,
            max_tokens=200,
            system=JUDGE_SYS,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content
                       if getattr(b, "type", "") == "text").strip()
                       
    elif provider == "google":
        from google.genai import types as genai_types
        from pydantic import BaseModel, Field

        # Define your exact schema. Gemini will adhere strictly to this.
        class JudgeDecision(BaseModel):
            merge: bool = Field(description="True if the click set focus exactly on the input field")
            reason: str = Field(description="One short sentence explaining the decision")
            merged_thought: str = Field(
                description="When merge=true, a single concise FARA-style "
                            "reasoning sentence for the canonical type action: "
                            "describe entering/inputting/typing the target text "
                            "into the relevant field; preserve goal/context; do "
                            "NOT mention clicking, \"already focused\", or a "
                            "prior action. "
                            "When merge=false, empty string."
            )

        resp = client.models.generate_content(
            model=model,
            contents=user,
            config=genai_types.GenerateContentConfig(
                system_instruction=JUDGE_SYS,
                response_mime_type="application/json",
                response_schema=JudgeDecision, # Enforces strict structured output
                max_output_tokens=512,
            ),
        )
        parsed = getattr(resp, "parsed", None)
        if parsed is not None:
            if hasattr(parsed, "model_dump"):
                return parsed.model_dump()
            if hasattr(parsed, "dict"):
                return parsed.dict()
            if isinstance(parsed, dict):
                return parsed
        text = (resp.text or "").strip()
    else:
        raise ValueError(f"unknown provider: {provider}")

    try:
        return _extract_json_object(text)
    except json.JSONDecodeError:
        return {"merge": None, "reason": f"PARSE_FAIL: {text[:200]}"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/gpfs/scrubbed/reza/MolmoWeb-SyntheticTrajs/data_filtered")
    p.add_argument("--out_dir", default="/gpfs/scrubbed/reza/fara/viz/click_type_merge")
    p.add_argument("--num_pairs", type=int, default=100)
    p.add_argument("--rows_per_file", type=int, default=200)
    p.add_argument("--max_files", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--provider", choices=["anthropic", "google"], default="anthropic",
                   help="LLM provider. 'google' = Gemini via google-genai SDK.")
    p.add_argument("--model", default=None,
                   help="Model id. Defaults: claude-haiku-4-5 (anthropic), "
                        "gemini-2.5-flash (google).")
    p.add_argument("--skip_llm", action="store_true",
                   help="Skip LLM judge; only produce HTML for manual eyeballing.")
    args = p.parse_args()
    if args.model is None:
        args.model = {"anthropic": "claude-haiku-4-5",
                      "google":    "gemini-2.5-flash"}[args.provider]

    rng = random.Random(args.seed)
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.parquet")))[: args.max_files]
    if not files:
        sys.exit(f"No parquet files in {args.data_dir}")

    pool = []
    for fp in files:
        fname = os.path.basename(fp)
        df = pd.read_parquet(fp).head(args.rows_per_file)
        for ri in range(len(df)):
            traj = _parse_traj(df.iloc[ri]["trajectory"])
            for i, ck, tk, click, typ in _find_pairs(traj):
                pool.append((fname, ri, i, ck, tk, click, typ))
    print(f"Found {len(pool)} click->type pairs across {len(files)} file(s)")
    if not pool:
        sys.exit("No pairs found; nothing to do")
    rng.shuffle(pool)
    pool = pool[: args.num_pairs]
    print(f"Sampled {len(pool)} pairs")

    client = None
    if not args.skip_llm:
        if args.provider == "anthropic":
            try:
                import anthropic
                client = anthropic.Anthropic()
            except ImportError:
                print("anthropic SDK not installed; pip install anthropic. "
                      "Continuing without LLM judge.")
        elif args.provider == "google":
            try:
                from google import genai
                client = genai.Client()
            except ImportError:
                print("google-genai SDK not installed; pip install google-genai. "
                      "Continuing without LLM judge.")

    # Load screenshots for the sampled pairs, grouped by file.
    by_file = {}
    for k, item in enumerate(pool):
        by_file.setdefault(item[0], []).append((k, item))
    screenshots = {}  # k -> (click_img, type_img)
    for fname, items in by_file.items():
        df = pd.read_parquet(os.path.join(args.data_dir, fname),
                             columns=["images"]).head(args.rows_per_file)
        for k, (_, ri, i, _, _, _, _) in items:
            imgs = df.iloc[ri]["images"]
            try:
                screenshots[k] = (_load_img(imgs[i]), _load_img(imgs[i + 1]))
            except Exception:
                screenshots[k] = (None, None)

    os.makedirs(args.out_dir, exist_ok=True)
    records = []
    for k, (fname, ri, i, ck, tk, click, typ) in enumerate(pool):
        click_bbox = click["args"].get("bbox") or click["args"].get("coordinate")
        type_bbox = typ["args"].get("bbox") or typ["args"].get("coordinate")
        rec = {
            "file": fname, "traj_idx": int(ri),
            "click_step_key": str(ck), "type_step_key": str(tk),
            "click_action_str": click["action_str"],
            "click_thought": click["thought"],
            "click_args": click["args"],
            "type_action_str": typ["action_str"],
            "type_thought": typ["thought"],
            "type_args": typ["args"],
            "click_bbox": click_bbox, "type_bbox": type_bbox,
        }
        if client is not None:
            rec["judge"] = _judge(client, args.model, args.provider, click, typ)
            print(f"[{k+1}/{len(pool)}] {rec['judge']}")
        else:
            rec["judge"] = None
        records.append((rec, *screenshots.get(k, (None, None))))

    blocks = []
    for idx, (rec, click_img, type_img) in enumerate(records):
        click_html = (
            f"<div style='flex:1;'><div><b>pre-click</b> (red=click bbox)</div>"
            f"<img src='data:image/jpeg;base64,{_img_b64(_draw_bbox(click_img, rec['click_bbox']))}' "
            f"style='max-width:100%;'/></div>"
            if click_img is not None else "<div style='flex:1;'><i>no image</i></div>"
        )
        type_html = (
            f"<div style='flex:1;'><div><b>pre-type</b> (green=type bbox if any)</div>"
            f"<img src='data:image/jpeg;base64,{_img_b64(_draw_bbox(type_img, rec['type_bbox'], color='green'))}' "
            f"style='max-width:100%;'/></div>"
            if type_img is not None else "<div style='flex:1;'><i>no image</i></div>"
        )
        j = rec.get("judge") or {}
        merge = j.get("merge")
        reason = j.get("reason", "")
        merged_thought = j.get("merged_thought", "")
        badge_bg = {True: "#080", False: "#a00"}.get(merge, "#888")
        badge_text = {True: "MERGE", False: "NO-MERGE"}.get(merge, "—")
        blocks.append(
            f"<section style='border:2px solid #333;border-radius:8px;padding:12px;margin:24px 0;background:#fff;'>"
            f"<h3 style='margin:0 0 8px 0;'>#{idx} &middot; {_html.escape(rec['file'])} &middot; "
            f"traj={rec['traj_idx']} &middot; steps {rec['click_step_key']}&rarr;{rec['type_step_key']} "
            f"<span style='background:{badge_bg};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;'>{badge_text}</span></h3>"
            f"<div style='display:flex;gap:12px;'>{click_html}{type_html}</div>"
            f"<div style='font-family:monospace;font-size:12px;margin-top:8px;'>"
            f"<details open><summary><b>Click</b> {_html.escape(rec['click_action_str'])}</summary>"
            f"<pre style='white-space:pre-wrap;'>{_html.escape(rec['click_thought'])}</pre></details>"
            f"<details open><summary><b>Type</b> {_html.escape(rec['type_action_str'])}</summary>"
            f"<pre style='white-space:pre-wrap;'>{_html.escape(rec['type_thought'])}</pre></details>"
            f"<b>Judge reason:</b> {_html.escape(reason)}"
            + (f"<br/><b>Merged thought:</b> "
               f"<pre style='white-space:pre-wrap;'>{_html.escape(merged_thought)}</pre>"
               if merged_thought else "")
            + f"</div></section>"
        )

    html = (
        "<html><body style='background:#fafafa;font-family:sans-serif;'>"
        "<style>.ct-report, .ct-report * { color:#000 !important; }</style>"
        f"<div class='ct-report'><h2>Click&rarr;Type pair eyeball ({len(records)} pairs)</h2>"
        + "".join(blocks) + "</div></body></html>"
    )
    html_path = os.path.join(args.out_dir, "report.html")
    with open(html_path, "w") as f:
        f.write(html)
    jsonl_path = os.path.join(args.out_dir, "decisions.jsonl")
    with open(jsonl_path, "w") as f:
        for rec, _, _ in records:
            f.write(json.dumps(rec, default=str) + "\n")

    if client is not None:
        m = sum(1 for r, _, _ in records if (r.get("judge") or {}).get("merge") is True)
        n = sum(1 for r, _, _ in records if (r.get("judge") or {}).get("merge") is False)
        b = sum(1 for r, _, _ in records if (r.get("judge") or {}).get("merge") is None)
        print(f"Judge: merge={m}  no-merge={n}  parse-fail={b}")
    print(f"HTML : {html_path}")
    print(f"JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
