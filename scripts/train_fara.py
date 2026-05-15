"""
SFT training script for Fara-7B.

Supports:
  * Full fine-tune and LoRA (--lora)
  * Multi-scale patches and single-scale (--no_multiscale)
  * Single-GPU, and multi-GPU via `accelerate launch` / `torchrun`

Data assumptions (MolmoWeb-SyntheticTrajs parquet):
  * Columns include: sample_id, instruction, trajectory, images
  * `images`    : list of dicts with a "bytes" key (PNG-encoded frames)
  * `trajectory`: list of step dicts. This script assumes each step is a dict
    with fields that can be serialized to text (e.g. {"thought", "action"}).
    ADJUST `row_to_messages` below if your schema differs.

Install deps in the training env (NOT the fara_webeval env):
    pip install "trl>=0.12" "transformers>=4.46" peft accelerate datasets \
                pyarrow bitsandbytes
"""

import argparse
import io
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import hashlib

# Prevent DeepSpeed import-time CUDA op probing from failing when using FSDP-only training.
os.environ.setdefault("DS_IGNORE_CUDA_DETECTION", "1")

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from PIL import Image
from transformers import Qwen2_5_VLProcessor
from transformers.trainer_callback import TrainerCallback
from fara.modeling.processing_qwen2_5_vl import FaraProcessor

from fara.modeling.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from fara.modeling.image_processing_qwen2_vl import Qwen2VLImageProcessor
from fara._prompts import get_computer_use_system_prompt
from fara.training_utils import is_main_process, log, setup_logging

setup_logging()


# ---------------------------------------------------------------------------
# Production-aligned constants (mirror src/fara/fara_agent.py FaraAgent)
# ---------------------------------------------------------------------------

MLM_PROCESSOR_IM_CFG = {
    "min_pixels": 3136,
    "max_pixels": 12845056,
    "patch_size": 14,
    "merge_size": 2,
}
USER_MESSAGE = "Here is the next screenshot. Think about what to do next."
FN_CALL_TEMPLATE_NAME = "default"
TOOL_NAME = "computer_use"   # the only function registered in fara's <tools> block
MAX_URL_LENGTH = 100         # mirrors FaraAgent.MAX_URL_LENGTH


# ---------------------------------------------------------------------------
# Dataset adapter: parquet row -> chat messages
# ---------------------------------------------------------------------------

def _filter_cache_key(args: Any) -> str:
    domains = sorted(d.strip().lower() for d in args.allowed_domains.split(",") if d.strip())
    key_str = f"{args.data_path}|{','.join(domains)}|{args.domain_filter_mode}"
    return hashlib.md5(key_str.encode()).hexdigest()[:12]


def _decode_image(img_entry: Any) -> Image.Image:
    """Decode one image entry. Schema-tolerant."""
    if isinstance(img_entry, Image.Image):
        return img_entry.convert("RGB")
    if isinstance(img_entry, dict) and "bytes" in img_entry and img_entry["bytes"] is not None:
        raw = img_entry["bytes"]
    elif isinstance(img_entry, dict) and "path" in img_entry and img_entry["path"]:
        return Image.open(img_entry["path"]).convert("RGB")
    elif isinstance(img_entry, (bytes, bytearray)):
        raw = bytes(img_entry)
    else:
        raise TypeError(f"Unexpected image entry type: {type(img_entry)}")
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _extract_instruction_text(raw: Any) -> str:
    """MolmoWeb stores `instruction` as a JSON dict with high/mid/low_level keys.
    Train on `high_level` only — closest to inference-time user phrasing; the
    other levels leak the plan into the prompt.
    """
    if isinstance(raw, dict):
        d = raw
    elif isinstance(raw, str):
        s = raw.strip()
        try:
            d = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return s
        if not isinstance(d, dict):
            return s
    else:
        return str(raw)
    for key in ("high_level", "instruction", "low_level", "task", "goal"):
        v = d.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return json.dumps(d)


def _truncate_to_decision_point(
    messages: List[Dict[str, Any]],
    images: List[Any],
    t: int,
    max_n_images: int,
) -> tuple:
    """Slice the trajectory to end at assistant turn `t`, keeping images
    only for user turns in [t - max_n_images + 1, t]. Older user turns
    keep their text but lose their image placeholder.
    """
    end_idx = 2 * t + 3
    truncated = messages[:end_idx]
    keep_image_start = max(0, t - max_n_images + 1)

    new_messages: List[Dict[str, Any]] = []
    new_images: List[Any] = []
    user_turn_idx = 0
    img_idx = 0
    for msg in truncated:
        if msg["role"] == "user":
            has_image = any(isinstance(c, dict) and c.get("type") == "image"
                            for c in msg["content"])
            if user_turn_idx >= keep_image_start:
                new_messages.append(msg)
                if has_image:
                    new_images.append(images[img_idx])
            else:
                text_only = [c for c in msg["content"]
                             if not (isinstance(c, dict) and c.get("type") == "image")]
                new_messages.append({"role": "user", "content": text_only})
            if has_image:
                img_idx += 1
            user_turn_idx += 1
        else:
            new_messages.append(msg)

    return new_messages, new_images


def _build_labels(
    input_ids: List[int],
    assistant_header_ids: List[int],
    turn_end_ids: List[int],
    last_only: bool = False,
) -> List[int]:
    """Mask all tokens except assistant completions (-100). If `last_only`,
    unmask only the final assistant span (decision-point training).
    """
    labels = [-100] * len(input_ids)
    n = len(input_ids)
    hdr, end = assistant_header_ids, turn_end_ids

    spans = []
    i = 0
    while i <= n - len(hdr):
        if input_ids[i : i + len(hdr)] == hdr:
            start = i + len(hdr)
            j = start
            while j <= n - len(end):
                if input_ids[j : j + len(end)] == end:
                    break
                j += 1
            stop = min(j + len(end), n)
            spans.append((start, stop))
            i = stop
        else:
            i += 1

    if last_only and spans:
        spans = spans[-1:]

    for start, stop in spans:
        for k in range(start, stop):
            labels[k] = input_ids[k]
    return labels


def _is_blank_image_entry(img_entry: Any, max_var: float = 0.0) -> bool:
    """Return True if the image is perfectly uniform (variance == 0).
    PNG is lossless so a true about:blank viewport has exactly zero variance.
    """
    try:
        arr = np.asarray(_decode_image(img_entry), dtype=np.float32)
    except Exception:
        return False
    return float(arr.var()) <= max_var


# Map dataset action names to fara's `computer_use` action enum
# (src/fara/_prompts.py FaraComputerUse.parameters.properties.action.enum).
_ACTION_ALIASES = {
    # clicks
    "click":          "left_click",
    "left_click":     "left_click",
    # typing / keys
    "type":           "type",
    "keyboard_type":  "type",
    "key":            "key",
    "keyboard_press": "key",
    "press":          "key",
    # mouse movement / scrolling
    "mouse_move":     "mouse_move",
    "scroll":         "scroll",
    # navigation
    "goto":           "visit_url",
    "visit_url":      "visit_url",
    "web_search":     "web_search",
    "history_back":   "history_back",
    # misc
    "wait":           "wait",
    "terminate":      "terminate",
    "pause_and_memorize_fact": "pause_and_memorize_fact",
}

# Argument keys fara's `computer_use` tool recognizes. Anything else gets dropped
# from training assistant messages so we don't bake non-schema fields (bid, bbox,
# node_properties, ...) into the model.
_FARA_ARG_KEYS = {
    "action", "coordinate", "keys", "text",
    "press_enter", "delete_existing_text",
    "pixels", "url", "query", "fact", "time", "status",
}


def _normalize_action_name(name: Any) -> Any:
    if not isinstance(name, str):
        return name
    normalized = name.strip().lower()
    return _ACTION_ALIASES.get(normalized, normalized)


def _bbox_xywh_to_coordinate(bbox: Any) -> Optional[List[int]]:
    """Convert (x, y, w, h) bounding box to a single (cx, cy) center point."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x, y, w, h = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    return [round(x + w / 2.0), round(y + h / 2.0)]


def _normalize_scroll_pixels(raw_args: Dict[str, Any]) -> Optional[int]:
    """Map the dataset's scroll magnitude into fara's `pixels` arg.

    Dataset convention (from action_description): positive `delta_y` = scroll
    DOWN. Fara convention (FaraComputerUse.parameters): positive `pixels` =
    scroll UP. So pixels = -delta_y. Horizontal `delta_x` is dropped (fara has
    no separate horizontal-scroll arg).
    """
    for key in ("delta_y", "scroll_y", "dy"):
        v = raw_args.get(key)
        if isinstance(v, (int, float)):
            return -int(round(float(v)))
    # Pass-through if the dataset already uses fara's own field name.
    v = raw_args.get("pixels")
    if isinstance(v, (int, float)):
        return int(round(float(v)))
    return None


def _coerce_to_fara_args(action_name: Optional[str], raw_args: Dict[str, Any]) -> Dict[str, Any]:
    """Build a fara-schema-compatible arguments dict from arbitrary raw args.

    * Pulls `coordinate` out of `bbox` (x,y,w,h) when needed.
    * Maps the dataset's `delta_y` to fara's `pixels` for scroll (with sign flip).
    * Coerces `keys` from string to list.
    * Drops any keys not in the fara schema so we don't pollute training.
    """
    args: Dict[str, Any] = {}
    if action_name:
        args["action"] = action_name

    # Coordinate: prefer explicit, else derive from bbox center.
    if action_name in {"left_click", "mouse_move", "type"}:
        coord = raw_args.get("coordinate")
        if (not isinstance(coord, (list, tuple)) or len(coord) != 2) and "bbox" in raw_args:
            coord = _bbox_xywh_to_coordinate(raw_args["bbox"])
        if isinstance(coord, (list, tuple)) and len(coord) == 2:
            args["coordinate"] = [round(float(coord[0])), round(float(coord[1]))]

    # Scroll: derive `pixels` from delta_y (with sign flip).
    if action_name == "scroll":
        pixels = _normalize_scroll_pixels(raw_args)
        if pixels is not None:
            args["pixels"] = pixels

    # Type: extract trailing newline -> press_enter. Datasets often encode
    # "type then submit" by appending '\n' to the text; fara's runtime expects
    # an explicit `press_enter` boolean (defaulting True if missing, which is
    # surprising — so we always set it explicitly).
    if action_name == "type":
        text = raw_args.get("text")
        if isinstance(text, str):
            stripped = text.rstrip("\r\n")
            args["text"] = stripped
            args["press_enter"] = stripped != text  # True iff we removed a newline
        if "delete_existing_text" in raw_args:
            args["delete_existing_text"] = bool(raw_args["delete_existing_text"])

    # `keys` should be a list per the schema; coerce singular `key` and string forms.
    if action_name == "key":
        keys = raw_args.get("keys")
        if keys is None:
            keys = raw_args.get("key")           # dataset sometimes uses singular `key`
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(keys, list):
            args["keys"] = keys

    # Terminate: ensure `status` is always set (schema requires it).
    if action_name == "terminate":
        status = raw_args.get("status")
        args["status"] = status if status in ("success", "failure") else "success"

    # Carry through any other recognized fields verbatim.
    for k, v in raw_args.items():
        if k in _FARA_ARG_KEYS and k not in args:
            args[k] = v

    return args


def _format_assistant_message(step: Any) -> Optional[str]:
    """Build a production-format assistant turn from a trajectory step.

    Returns None when the step's action has no fara-compatible mapping
    (e.g. `noop`, `dblclick`, `tab_focus`). The caller drops the whole
    trajectory if any step here returns None — keeps causality intact.

    Output mirrors what fara emits at inference (parsed by
    `FaraAgent._parse_thoughts_and_action`):

        <thought text>
        <tool_call>
        {"name": "computer_use", "arguments": {"action": <name>, ...}}
        </tool_call>

    The thought is a free-text prefix, NOT inside arguments — fara's runtime
    extracts it from the leading text and only then adds it to arguments.
    """
    if isinstance(step, str):
        return step

    if not isinstance(step, dict):
        return str(step)

    thought: Optional[str] = None
    action_name: Optional[str] = None
    action_args: Dict[str, Any] = {}

    # Molmo/WebVoyager-style nested action payload.
    action_block = step.get("action") if isinstance(step.get("action"), dict) else None
    if action_block is not None:
        ao = action_block.get("action_output") if isinstance(action_block.get("action_output"), dict) else {}
        if isinstance(ao, dict):
            thought = ao.get("thought")
            action_name = _normalize_action_name(ao.get("action_name"))
            inner_action = ao.get("action")
            if isinstance(inner_action, dict):
                action_args.update(inner_action)
        # Some datasets store the args dict on the action_block directly.
        for key in ("arguments", "args"):
            maybe_args = action_block.get(key)
            if isinstance(maybe_args, dict):
                action_args.update(maybe_args)

    # Flat-dict fallback: {"thought": ..., "action": "<name>" or {...}}.
    if action_name is None:
        if isinstance(step.get("action"), str):
            action_name = _normalize_action_name(step["action"])
        elif isinstance(step.get("action_name"), str):
            action_name = _normalize_action_name(step["action_name"])
    if thought is None and isinstance(step.get("thought"), str):
        thought = step["thought"]
    if not action_args and isinstance(step.get("arguments"), dict):
        action_args.update(step["arguments"])

    # If the inner action dict carried its own action name, prefer it (matches
    # production where arguments["action"] is the sub-action of computer_use).
    if action_name is None and isinstance(action_args.get("action"), str):
        action_name = _normalize_action_name(action_args["action"])

    # `send_msg_to_user` isn't a fara action, but its semantics are "report the
    # final answer and stop." Production fara reads `final_answer = thoughts`
    # right before a `terminate` (fara_agent.py:513-515). So fold the message
    # into the thought and emit terminate(status="success").
    if action_name == "send_msg_to_user":
        msg = (action_args.get("msg")
               or action_args.get("message")
               or action_args.get("text"))
        if isinstance(msg, str) and msg.strip():
            # Strip MolmoWeb's [ANSWER] / [EXIT] / [FAILURE] bracket tags — they're
            # bookkeeping markers, not part of the user-facing answer.
            cleaned = re.sub(r"^\s*\[(?:ANSWER|EXIT|FAILURE)\]\s*", "", msg.strip())
            base = (thought or "").strip()
            thought = (base + "\n\n" + cleaned).strip() if base else cleaned
        action_name = "terminate"
        action_args = {}

    # `browser_nav` covers history navigation. Only `go_back` maps to a fara
    # action (history_back). Other nav_types (tab_focus, ...) have no fara
    # equivalent and signal that the trajectory should be dropped.
    if action_name == "browser_nav":
        nav_type = action_args.get("nav_type")
        if nav_type == "go_back":
            action_name = "history_back"
            action_args = {}
        else:
            return None    # tab_focus / others -> drop trajectory

    # Actions with no fara mapping. Returning None tells the caller to drop the
    # whole trajectory so we don't introduce unexplained state transitions.
    if action_name in {"noop", "dblclick"}:
        return None

    # Coerce raw args to fara's schema: bbox -> coordinate, drop non-schema keys,
    # remap action names (e.g. "click" -> "left_click").
    fara_args = _coerce_to_fara_args(action_name, action_args)

    if not fara_args.get("action"):
        return None        # unrecognized; drop trajectory

    tool_call_obj = {"name": TOOL_NAME, "arguments": fara_args}
    tool_call_block = "<tool_call>\n" + json.dumps(tool_call_obj, ensure_ascii=False) + "\n</tool_call>"

    if thought:
        return f"{thought.strip()}\n{tool_call_block}"
    return tool_call_block


def _parse_trajectory(raw_trajectory: Any) -> List[Any]:
    """Normalize trajectory payload into an ordered list of step objects."""
    traj = raw_trajectory

    if isinstance(traj, str):
        try:
            traj = json.loads(traj)
        except Exception:
            return []

    if isinstance(traj, list):
        return traj

    if isinstance(traj, dict):
        # Many datasets store steps as {"1": {...}, "2": {...}}.
        if all(isinstance(k, str) and k.isdigit() for k in traj.keys()):
            return [traj[k] for k in sorted(traj.keys(), key=lambda x: int(x))]
        return list(traj.values())

    return []


def _normalize_url(u: Any) -> Optional[str]:
    if not isinstance(u, str):
        return None
    u = u.strip()
    if not u:
        return None
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u):
        u = "https://" + u
    return u


def _extract_domain(u: str) -> Optional[str]:
    try:
        netloc = urlparse(u).netloc.lower().strip()
    except Exception:
        return None
    if not netloc:
        return None
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc or None


def _domain_allowed(domain: str, allowed_domains: List[str]) -> bool:
    # Allow exact match and subdomains (e.g., export.arxiv.org for arxiv.org).
    for allowed in allowed_domains:
        if domain == allowed or domain.endswith("." + allowed):
            return True
    return False


def _extract_step_urls(step: Any) -> List[str]:
    if not isinstance(step, dict):
        return []

    urls: List[str] = []

    # Browser observation URLs.
    other_obs = step.get("other_obs", {}) if isinstance(step.get("other_obs", {}), dict) else {}
    if isinstance(other_obs.get("url"), str):
        urls.append(other_obs["url"])
    open_pages_urls = other_obs.get("open_pages_urls", [])
    if isinstance(open_pages_urls, list):
        urls.extend([u for u in open_pages_urls if isinstance(u, str)])

    # Nested action URL.
    action_block = step.get("action", {}) if isinstance(step.get("action", {}), dict) else {}
    action_output = action_block.get("action_output", {}) if isinstance(action_block.get("action_output", {}), dict) else {}
    action_obj = action_output.get("action", {}) if isinstance(action_output.get("action", {}), dict) else {}
    if isinstance(action_obj.get("url"), str):
        urls.append(action_obj["url"])

    # Parse strings like goto(url='...') and visit_url(url='...').
    action_str = action_block.get("action_str")
    if isinstance(action_str, str):
        m = re.search(r"(?:goto|visit_url)\s*\(\s*url\s*=\s*['\"]([^'\"]+)['\"]", action_str)
        if m:
            urls.append(m.group(1))

    return urls


def _extract_row_domains(raw_trajectory: Any) -> List[str]:
    steps = _parse_trajectory(raw_trajectory)
    domains: List[str] = []
    for step in steps:
        for raw_url in _extract_step_urls(step):
            normalized = _normalize_url(raw_url)
            if not normalized:
                continue
            domain = _extract_domain(normalized)
            if domain:
                domains.append(domain)
    return domains


def _validate_image_step_alignment(images: List[Any], steps: List[Any], sample_id: Any = None) -> None:
    """Check that image path aligns with step screenshot for each paired frame."""
    mode = os.getenv("FARA_VALIDATE_IMAGE_STEP_ALIGNMENT", "strict").strip().lower()
    if mode in {"0", "off", "false", "none"}:
        return

    mismatches = []
    for i, (img, step) in enumerate(zip(images, steps)):
        if not isinstance(img, dict) or not isinstance(step, dict):
            continue
        img_path = img.get("path")
        screenshot = step.get("screenshot")
        if not img_path or not screenshot:
            continue

        img_name = os.path.basename(str(img_path))
        shot_name = os.path.basename(str(screenshot))
        if img_name != shot_name:
            mismatches.append((i, img_name, shot_name))

    if not mismatches:
        return

    sid = "unknown" if sample_id is None else str(sample_id)
    detail = ", ".join([f"idx={i}: image='{ip}' vs step='{sp}'" for i, ip, sp in mismatches[:5]])
    msg = f"[fara-train] image/trajectory mismatch sample_id={sid}: {detail}"

    if mode in {"strict", "error", "raise"}:
        raise ValueError(msg)
    log(msg)


def _get_step_url(step: Any) -> Optional[str]:
    """Extract the active URL for a step (matches what fara reads at runtime)."""
    if not isinstance(step, dict):
        return None
    other_obs = step.get("other_obs") if isinstance(step.get("other_obs"), dict) else {}
    url = other_obs.get("url") if isinstance(other_obs, dict) else None
    if isinstance(url, str) and url.strip():
        return url.strip()
    # Fallback to the first URL of any kind we can find for this step.
    urls = _extract_step_urls(step)
    return urls[0] if urls else None


def _trim_url(url: str, max_len: int = MAX_URL_LENGTH) -> str:
    if len(url) <= max_len:
        return url
    return url[: max_len - 3] + "..."


def _build_system_prompt_for_image(image_for_dims: Any) -> str:
    """Build the production system prompt (with <tools> block) using the
    image's dimensions so display_width_px / display_height_px match what the
    runtime would produce for the same screenshot.
    """
    pil = _decode_image(image_for_dims)
    info = get_computer_use_system_prompt(
        pil,
        MLM_PROCESSOR_IM_CFG,
        include_input_text_key_args=True,
        fn_call_template=FN_CALL_TEMPLATE_NAME,
    )
    return _extract_system_prompt_text(info)


def row_to_messages(row: Dict[str, Any], system_prompt_text: Optional[str] = None) -> Dict[str, Any]:
    """Convert one parquet row to Qwen2.5-VL chat messages + raw image entries.

    Mirrors fara's production message layout (see src/fara/fara_agent.py):
      * System: full <tools>-aware prompt from get_computer_use_system_prompt,
        sized to the first screenshot's dims.
      * Turn 0:        user(image, <task instruction>)
      * Turn i > 0:    user(image, "Current URL: <trimmed_url>\\n<USER_MESSAGE>")
      * Each assistant: <thought>\\n<tool_call>{"name":"computer_use",...}</tool_call>

    Returns:
        {"messages": [...], "images": [<raw image entries>]}
    """
    sample_id = row.get("sample_id")
    instruction = _extract_instruction_text(row.get("instruction", ""))
    images_raw: List[Any] = row["images"]
    trajectory_raw = row.get("trajectory", [])

    # Keep raw image payloads in the dataset and decode in the collator.
    # This avoids serializing PIL objects through datasets map workers.
    images = list(images_raw)
    steps = _parse_trajectory(trajectory_raw)

    # Align images and steps. If mismatched, truncate to the shorter.
    n = min(len(images), len(steps))
    images = images[:n]
    steps = steps[:n]

    # Optional sanity check for datasets that provide image path + step screenshot keys.
    _validate_image_step_alignment(images, steps, sample_id=sample_id)

    if system_prompt_text is None and images:
        try:
            system_prompt_text = _build_system_prompt_for_image(images[0])
        except Exception as e:
            log(f"[fara-train] failed to build production system prompt "
                  f"(sample_id={sample_id}): {e}; falling back to generic.")
    if not system_prompt_text:
        system_prompt_text = (
            "You are a web agent. Given a screenshot and an instruction, "
            "produce the next action to take."
        )

    # Format every assistant turn first. If ANY step has no fara mapping, drop
    # the whole trajectory by returning empty messages/images — easier to
    # filter post-map than to truncate / skip and risk causality breaks.
    formatted_steps: List[str] = []
    for step in steps:
        text = _format_assistant_message(step)
        if text is None:
            return {"messages": [], "images": []}
        formatted_steps.append(text)

    messages: List[Dict[str, Any]] = [
        {"role": "system",
         "content": [{"type": "text", "text": system_prompt_text}]},
    ]
    for i, (step, assistant_text) in enumerate(zip(steps, formatted_steps)):
        if i == 0:
            user_text = instruction
        else:
            url = _get_step_url(step)
            url_prefix = f"Current URL: {_trim_url(url)}\n" if url else ""
            user_text = f"{url_prefix}{USER_MESSAGE}"

        messages.append({
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": user_text}],
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text}],
        })
    return {"messages": messages, "images": images}


def _extract_system_prompt_text(prompt_info: Dict[str, Any]) -> str:
    """Flatten system message text from get_computer_use_system_prompt output."""
    chunks: List[str] = []
    for msg in prompt_info.get("conversation", []):
        if msg.get("role") != "system":
            continue
        for item in msg.get("content", []):
            text = item.get("text") if isinstance(item, dict) else None
            if text:
                chunks.append(text)
    return "\n".join(chunks).strip()


# ---------------------------------------------------------------------------
# Dataset: per-sample processing (decoding, trajectory diff, tokenization)
# Collator: pad + stack only
# ---------------------------------------------------------------------------

ASSISTANT_HEADER = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>"


def _count_final_tokens(image_grid_thw: Any, merge_size: int) -> int:
    if image_grid_thw is None:
        return 0
    total = 0
    for grid in image_grid_thw:
        if hasattr(grid, "tolist"):
            grid = grid.tolist()
        total += int(np.prod(grid)) // (merge_size ** 2)
    return total


class PatchStatsSFTTrainer:
    def _init_patch_stats(self) -> None:
        self._patch_baseline_tokens = 0
        self._patch_final_tokens = 0
        self._patch_samples = 0

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        from transformers import Trainer
        decay_names = set(self.get_decay_parameter_names(self.model))

        def is_vision(name: str) -> bool:
            return name.startswith("visual.") or ".visual." in name

        groups = {
            ("vision", True):  {"params": [], "lr": self.args.vision_learning_rate,
                                "weight_decay": self.args.weight_decay},
            ("vision", False): {"params": [], "lr": self.args.vision_learning_rate,
                                "weight_decay": 0.0},
            ("llm", True):     {"params": [], "lr": self.args.learning_rate,
                                "weight_decay": self.args.weight_decay},
            ("llm", False):    {"params": [], "lr": self.args.learning_rate,
                                "weight_decay": 0.0},
        }

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            kind = "vision" if is_vision(n) else "llm"
            decays = n in decay_names
            groups[(kind, decays)]["params"].append(p)
        
        param_groups = [g for g in groups.values() if g["params"]]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        optimizer_kwargs.pop("lr", None)
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)

        # Optional: log group sizes once
        if is_main_process():
            for g in param_groups:
                n_params = sum(p.numel() for p in g["params"])
                log(f"[fara-train] optim group lr={g['lr']:.2e} wd={g['weight_decay']} params={n_params:,}")

            # Warn when vision LR was set but no trainable vision params exist
            # (e.g., default LoRA only injects into LLM attn modules, or --freeze_vision).
            vision_lr = self.args.vision_learning_rate
            has_vision_group = any(
                g["lr"] == vision_lr and g["params"] for g in param_groups
            )
            if not has_vision_group and vision_lr != self.args.learning_rate:
                log(
                    f"[fara-train] vision_learning_rate={vision_lr:.2e} set but no "
                    f"trainable vision params found (likely default LoRA target_modules "
                    f"or --freeze_vision). Vision LR is a no-op for this run.",
                    level=logging.WARNING,
                )

        return self.optimizer

    def _accumulate_patch_stats(self, inputs: Dict[str, Any]) -> None:
        baseline = inputs.pop("patch_baseline_tokens", None)
        final = inputs.pop("patch_final_tokens", None)
        samples = inputs.pop("patch_sample_count", None)
        if baseline is None or final is None:
            return

        baseline_value = int(baseline.item()) if torch.is_tensor(baseline) else int(baseline)
        final_value = int(final.item()) if torch.is_tensor(final) else int(final)
        samples_value = int(samples.item()) if torch.is_tensor(samples) else int(samples or 0)

        stats = torch.tensor([baseline_value, final_value, samples_value], dtype=torch.long)

        # NOTE: make sure that this part won't cause NCCL errors in the future
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device = getattr(self.args, "device", torch.device("cpu"))
            stats = stats.to(device)
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
            stats = stats.cpu()

        self._patch_baseline_tokens += int(stats[0].item())
        self._patch_final_tokens += int(stats[1].item())
        self._patch_samples += int(stats[2].item())

    def training_step(self, model, inputs, num_items_in_batch=None):
        if isinstance(inputs, dict):
            self._accumulate_patch_stats(inputs)
        return super().training_step(model, inputs, num_items_in_batch)

    def log(self, logs, start_time=None):
        if getattr(self, "_patch_samples", 0) > 0 and getattr(self, "_patch_baseline_tokens", 0) > 0:
            logs = {} if logs is None else dict(logs)
            # Compute running means (average per sample)
            mean_baseline_tokens = self._patch_baseline_tokens / self._patch_samples if self._patch_samples > 0 else 0.0
            mean_final_tokens = self._patch_final_tokens / self._patch_samples if self._patch_samples > 0 else 0.0
            mean_tokens_saved = mean_baseline_tokens - mean_final_tokens
            reduction_pct = (mean_tokens_saved / mean_baseline_tokens) * 100.0 if mean_baseline_tokens > 0 else 0.0
            logs["patch_stats/mean_baseline_tokens_per_sample"] = mean_baseline_tokens
            logs["patch_stats/mean_final_tokens_per_sample"] = mean_final_tokens
            logs["patch_stats/mean_tokens_saved_per_sample"] = mean_tokens_saved
            logs["patch_stats/mean_reduction_pct"] = reduction_pct
            logs["patch_stats/samples"] = self._patch_samples
        return super().log(logs, start_time)


_COORD_RE = re.compile(r'"coordinate"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]')


def _extract_coord(text: str) -> Optional[List[float]]:
    if not isinstance(text, str):
        return None
    m = _COORD_RE.search(text)
    if not m:
        return None
    return [float(m.group(1)), float(m.group(2))]


def _last_assistant_text(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            parts = msg.get("content", [])
            if isinstance(parts, list):
                for c in parts:
                    if isinstance(c, dict) and c.get("type") == "text":
                        return c.get("text", "")
            if isinstance(parts, str):
                return parts
    return ""


def _first_user_text(messages: List[Dict[str, Any]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            parts = msg.get("content", [])
            if isinstance(parts, list):
                for c in parts:
                    if isinstance(c, dict) and c.get("type") == "text":
                        return c.get("text", "")
            if isinstance(parts, str):
                return parts
    return ""


_ACTION_RE = re.compile(r'"action"\s*:\s*"([^"]+)"')


def _extract_action(text: str) -> Optional[str]:
    m = _ACTION_RE.search(text)
    return m.group(1) if m else None


def _render_coord_image(
    image: "Image.Image",
    pred: Optional[List[float]],
    gt: Optional[List[float]],
) -> "Image.Image":
    from PIL import ImageDraw

    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    r = 12
    if gt is not None:
        x, y = gt
        draw.ellipse((x - r, y - r, x + r, y + r), outline="lime", width=4)
        draw.text((x + r + 4, y + r + 4), "gt", fill="lime")
    if pred is not None:
        x, y = pred
        draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=4)
        draw.line((x - r - 6, y, x + r + 6, y), fill="red", width=2)
        draw.line((x, y - r - 6, x, y + r + 6), fill="red", width=2)
        draw.text((x + r + 4, y - r - 14), "pred", fill="red")
    return img


class InferenceEvalCallback(TrainerCallback):
    """On each eval cycle, run model.generate on a fixed random subset of the
    val set and log the screenshots (with predicted vs ground-truth coords
    overlaid when present) to wandb.

    Same indices each eval -> easy to track per-sample evolution over time.
    """

    def __init__(
        self,
        eval_hf_dataset: Any,
        processor: Any,
        n_samples: int,
        max_n_images: int,
        max_new_tokens: int,
        seed: int,
    ) -> None:
        self.dataset = eval_hf_dataset
        self.processor = processor
        self.max_n_images = max_n_images
        self.max_new_tokens = max_new_tokens
        n = min(n_samples, len(eval_hf_dataset))
        rng = random.Random(seed)
        self.indices = rng.sample(range(len(eval_hf_dataset)), n) if n > 0 else []

    MAX_TURNS_PER_ROW = 10

    def _iter_turn_inputs(self, row: Dict[str, Any]):
        """For each decision point t in [0, min(n_turns, MAX_TURNS_PER_ROW)),
        yield (enc, last_screenshot, gt_text, t). The prompt ends with the
        assistant header so model.generate has to emit the next turn.
        """
        messages_full = row["messages"]
        pil_images_full = [_decode_image(img) for img in row["images"]]
        n_turns = (len(messages_full) - 1) // 2
        if n_turns <= 0:
            return
        n = min(n_turns, self.MAX_TURNS_PER_ROW)
        for t in range(n):
            messages, pil_images = _truncate_to_decision_point(
                messages_full, pil_images_full, t, max_n_images=self.max_n_images,
            )
            if not messages or messages[-1].get("role") != "assistant":
                continue
            if not pil_images:
                continue
            gt_text = _last_assistant_text(messages)
            prompt_messages = messages[:-1]
            enc = self.processor(
                messages=prompt_messages,
                images=pil_images,
                add_generation_prompt=True,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )
            yield enc, pil_images[-1], gt_text, t

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not self.indices:
            return
        try:
            import wandb
        except ImportError:
            return
        # if not is_main_process() or wandb.run is None:
            # return
        if is_main_process() and wandb.run is not None:
            device = next(model.parameters()).device
            was_training = model.training
            model.eval()
            import base64, html as _html
            from io import BytesIO
            task_sections: List[str] = []
            try:
                for idx in self.indices:
                    row = self.dataset[idx]
                    task_text = _first_user_text(row["messages"])
                    turn_blocks: List[str] = []
                    for enc, last_screenshot, gt_text, t in self._iter_turn_inputs(row):
                        inputs = {k: v.to(device) for k, v in enc.items()
                                if torch.is_tensor(v)}
                        with torch.no_grad():
                            out = model.generate(
                                **inputs,
                                max_new_tokens=self.max_new_tokens,
                                do_sample=False,
                                pad_token_id=self.processor.tokenizer.pad_token_id
                                    or self.processor.tokenizer.eos_token_id,
                            )
                        prompt_len = inputs["input_ids"].shape[1]
                        pred_text = self.processor.tokenizer.decode(
                            out[0, prompt_len:], skip_special_tokens=True,
                        )
                        pred_coord = _extract_coord(pred_text)
                        gt_coord = _extract_coord(gt_text)
                        pred_action = _extract_action(pred_text)
                        gt_action = _extract_action(gt_text)

                        rendered = _render_coord_image(last_screenshot, pred_coord, gt_coord)
                        buf = BytesIO()
                        rendered.save(buf, format="JPEG", quality=85)
                        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                        turn_blocks.append(
                            f'<div style="display:flex;gap:12px;margin:12px 0;'
                            f'border-bottom:1px dashed #bbb;padding-bottom:12px;">'
                            f'<img src="data:image/jpeg;base64,{b64}" '
                            f'style="max-width:50%;height:auto;object-fit:contain;"/>'
                            f'<div style="flex:1;font-family:monospace;font-size:12px;">'
                            f'<div><b>turn={t} step={state.global_step}</b></div>'
                            f'<div>pred_action={pred_action} pred_coord={pred_coord}</div>'
                            f'<div>gt_action={gt_action} gt_coord={gt_coord}</div>'
                            f'<details open><summary><b>PRED</b></summary>'
                            f'<pre style="white-space:pre-wrap;">{_html.escape(pred_text)}</pre></details>'
                            f'<details><summary><b>GT</b></summary>'
                            f'<pre style="white-space:pre-wrap;">{_html.escape(gt_text)}</pre></details>'
                            f'</div></div>'
                        )
                    if turn_blocks:
                        task_sections.append(
                            f'<section style="border:3px solid #333;border-radius:8px;'
                            f'background:#fff;padding:16px;margin:32px 0;">'
                            f'<h2 style="margin:0 0 8px 0;padding:8px 12px;'
                            f'background:#333;color:#fff;border-radius:4px;">'
                            f'TASK idx={idx}</h2>'
                            f'<div style="background:#eef;padding:10px 12px;border-radius:4px;'
                            f'margin:0 0 12px 0;font-family:monospace;font-size:13px;'
                            f'white-space:pre-wrap;">'
                            f'<b>User instruction:</b><br/>{_html.escape(task_text)}</div>'
                            + "".join(turn_blocks) +
                            f'</section>'
                        )
            finally:
                if was_training:
                    model.train()

            if task_sections:
                separator = (
                    '<hr style="border:0;border-top:6px double #000;margin:40px 0;"/>'
                )
                page = (
                    f'<html><body style="background:#fafafa;">'
                    f'<h3>Inference eval @ step {state.global_step}</h3>'
                    + separator.join(task_sections) +
                    f'</body></html>'
                )
                wandb.log(
                    {"eval/inf/samples": wandb.Html(page)},
                    step=wandb.run.step,
                )
 
        torch.distributed.barrier() if torch.distributed.is_available() and torch.distributed.is_initialized() else None


class FaraDataset(torch.utils.data.Dataset):
    """Per-sample processing runs here (in DataLoader workers).
    Returns fully tokenized tensors so the collator only pads/stacks.
    """

    def __init__(
        self,
        hf_dataset: Any,
        processor: Any,
        use_multiscale: bool = True,
        sampling_strategy: str = "decision_point",
        max_n_images: int = 3,
        max_seq_length: int = 8192,
    ) -> None:
        self.hf_dataset = hf_dataset
        self.processor: FaraProcessor = processor
        self.use_multiscale = use_multiscale
        self.sampling_strategy = sampling_strategy
        self.max_n_images = max_n_images
        self.max_seq_length = max_seq_length

        processor.image_processor.use_multiscale = use_multiscale
        tok = processor.tokenizer
        self._header_ids = tok.encode(ASSISTANT_HEADER, add_special_tokens=False)
        self._end_ids = tok.encode(TURN_END, add_special_tokens=False)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.hf_dataset[idx]
        messages = row["messages"]
        pil_images = [_decode_image(img) for img in row["images"]]

        if self.sampling_strategy == "decision_point":
            n_turns = (len(messages) - 1) // 2
            if n_turns > 0:
                t = random.randrange(n_turns)
                messages, pil_images = _truncate_to_decision_point(
                    messages, pil_images, t, max_n_images=self.max_n_images,
                )
        elif self.sampling_strategy == "full_trajectory":
            n_turns = (len(messages) - 1) // 2
            if n_turns > 0:
                messages, pil_images = _truncate_to_decision_point(
                    messages,
                    pil_images,
                    n_turns - 1,
                    max_n_images=self.max_n_images,
                )

        enc = self.processor(
            messages=messages,
            images=pil_images,
            padding=False,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        # Diagnostic logging for single-image sample detection with rank info
        # raw_image_count = len(row["images"])
        # decoded_image_count = len(pil_images)
        # has_pixel_values = "pixel_values" in enc
        # has_image_grid_thw = "image_grid_thw" in enc
        # rank = torch.distributed.get_rank() if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
        # print(f"[rank {rank}] [sample {idx}] raw_images={raw_image_count} decoded_images={decoded_image_count} has_pixel_values={has_pixel_values} has_image_grid_thw={has_image_grid_thw}")

        ids = enc["input_ids"][0].tolist()
        labels = _build_labels(
            ids, self._header_ids, self._end_ids,
            last_only=(self.sampling_strategy in {"decision_point", "full_trajectory"}),
        )

        out: Dict[str, torch.Tensor] = {
            "input_ids":      enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "labels":         torch.tensor(labels, dtype=torch.long),
        }
        if "pixel_values" in enc:
            out["pixel_values"]   = enc["pixel_values"]
            out["image_grid_thw"] = enc["image_grid_thw"]
        if "maybe_positions_multiscale" in enc:
            out["maybe_positions_multiscale"] = enc["maybe_positions_multiscale"]
            out["maybe_centers_multiscale"]   = enc["maybe_centers_multiscale"]

        image_status = enc.get("image_status")
        if image_status is None:
            active_images = pil_images
        else:
            active_images = [
                image for i, image in enumerate(pil_images)
                if (image_status[i] if i < len(image_status) else "active") in {"active", "mid_dup"}
            ]

        image_processor = self.processor.image_processor
        baseline_patches = sum(
            image_processor.get_number_of_image_patches(
                image.height,
                image.width,
                images_kwargs={
                    "min_pixels": image_processor.min_pixels,
                    "max_pixels": image_processor.max_pixels,
                    "patch_size": image_processor.patch_size,
                    "merge_size": image_processor.merge_size,
                },
            )
            for image in active_images
        )
        baseline_tokens = baseline_patches // (image_processor.merge_size ** 2)
        final_tokens = _count_final_tokens(enc.get("image_grid_thw"), image_processor.merge_size)

        out["patch_baseline_tokens"] = torch.tensor(baseline_tokens, dtype=torch.long)
        out["patch_final_tokens"] = torch.tensor(final_tokens, dtype=torch.long)
        out["patch_sample_count"] = torch.tensor(1, dtype=torch.long)

        return out


@dataclass
class FaraCollator:
    """Pad variable-length sequences and concatenate vision tensors into a batch.
    All heavy processing (decoding, trajectory diff, tokenization) happens in
    FaraDataset.__getitem__ (DataLoader workers).
    """
    pad_id: int

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(ex["input_ids"].shape[0] for ex in batch)
        B = len(batch)

        input_ids     = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long)
        labels        = torch.full((B, max_len), -100, dtype=torch.long)

        for i, ex in enumerate(batch):
            L = ex["input_ids"].shape[0]
            input_ids[i, :L]      = ex["input_ids"]
            attention_mask[i, :L] = 1
            labels[i, :L]         = ex["labels"]

        out: Dict[str, torch.Tensor] = {
            "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
        }

        if any("pixel_values" in ex for ex in batch):
            out["pixel_values"]   = torch.cat([ex["pixel_values"]   for ex in batch if "pixel_values"   in ex])
            # print(f"Collator: concatenated pixel_values with shape {out['pixel_values'].shape}")
            out["image_grid_thw"] = torch.cat([ex["image_grid_thw"] for ex in batch if "image_grid_thw" in ex])
            # print(f"Collator: image_grid_thw {out['image_grid_thw']}")

            if any("maybe_positions_multiscale" in ex for ex in batch):
                out["maybe_positions_multiscale"] = torch.cat([ex["maybe_positions_multiscale"] for ex in batch if "maybe_positions_multiscale" in ex])
                out["maybe_centers_multiscale"]   = torch.cat([ex["maybe_centers_multiscale"]   for ex in batch if "maybe_centers_multiscale"   in ex])

        if any("patch_baseline_tokens" in ex for ex in batch):
            out["patch_baseline_tokens"] = torch.tensor(
                sum(int(ex["patch_baseline_tokens"].item()) for ex in batch if "patch_baseline_tokens" in ex),
                dtype=torch.long,
            )
            out["patch_final_tokens"] = torch.tensor(
                sum(int(ex["patch_final_tokens"].item()) for ex in batch if "patch_final_tokens" in ex),
                dtype=torch.long,
            )
            out["patch_sample_count"] = torch.tensor(
                sum(int(ex["patch_sample_count"].item()) for ex in batch if "patch_sample_count" in ex),
                dtype=torch.long,
            )

        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Data / model / output
    p.add_argument("--data_path", required=True,
                   help="Parquet file or directory of parquet files.")
    p.add_argument("--model_id", default="microsoft/Fara-7B")
    p.add_argument("--output_dir", required=True)

    # Training hyperparams
    p.add_argument("--per_device_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-5,
                   help="Base LR (used for LLM params).")
    p.add_argument("--vision_learning_rate", type=float, default=None,
                   help="LR for vision-encoder params (visual.*). "
                        "Overrides --vision_lr_ratio when set.")
    p.add_argument("--vision_lr_ratio", type=float, default=0.1,
                   help="Vision-encoder LR as a fraction of --learning_rate. "
                        "Used only when --vision_learning_rate is unset.")
    p.add_argument("--num_epochs", type=float, default=1.0)
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3,
                   help="Max number of checkpoints to keep on disk. "
                        "Older ones are deleted. Set to None/0 to keep all.")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap dataset size (useful for smoke tests).")
    p.add_argument("--shuffle_seed", type=int, default=42,
                   help="Seed for shuffling the dataset before max_samples / "
                        "train-val split. Set to a negative value to disable.")
    p.add_argument("--viz_dir", type=str, default=None,
                   help="If set, after build_dataset finishes, dump up to "
                        "--viz_max_rows post-processed rows to "
                        "<viz_dir>/build_dataset_sample.html for sanity checking.")
    p.add_argument("--viz_max_rows", type=int, default=50,
                   help="Number of rows to include in the build-time viz dump.")
    p.add_argument("--val_split_ratio", type=float, default=0.0,
                   help="Fraction of the dataset to hold out as a validation set "
                        "(e.g., 0.02). 0 disables validation.")
    p.add_argument("--val_max_samples", type=int, default=None,
                   help="Cap the held-out val set after splitting.")
    p.add_argument("--val_split_seed", type=int, default=42,
                   help="Seed used for the deterministic train/val split.")
    p.add_argument("--eval_steps", type=int, default=500,
                   help="Run eval every N steps (only when val set is enabled).")
    p.add_argument("--per_device_eval_batch_size", type=int, default=None,
                   help="Per-device eval batch size. Defaults to per_device_batch_size.")
    p.add_argument("--inf_eval_samples", type=int, default=16,
                   help="Number of random val samples to run inference on per eval "
                        "cycle. 0 disables. Logged to wandb under eval/inf/.")
    p.add_argument("--inf_eval_seed", type=int, default=0,
                   help="Seed for the fixed random subset used by inference eval, so "
                        "the same samples are compared across eval steps.")
    p.add_argument("--inf_eval_max_new_tokens", type=int, default=192,
                   help="max_new_tokens for inference eval generation.")
    p.add_argument(
        "--allowed_domains",
        default="",
        help=(
            "Comma-separated allowlist for training domains (e.g., 'arxiv.org,allrecipes.com'). "
            "Rows are filtered before training when set."
        ),
    )
    p.add_argument(
        "--domain_filter_mode",
        default="any",
        choices=["any", "strict"],
        help=(
            "Domain filtering mode when --allowed_domains is set: "
            "'any' keeps rows with at least one allowed domain in trajectory; "
            "'strict' keeps rows only if all observed domains are in allowlist."
        ),
    )
    p.add_argument(
        "--data_cache_root",
        default="/gpfs/scrubbed/reza/fara/data_cache",
        help=(
            "Directory for the post-domain-filter Dataset cache. "
            "Set to empty string to disable caching."
        ),
    )
    p.add_argument("--report_to", default="none",
                   help="Logging backend(s): 'none', 'wandb', or comma-separated values accepted by Transformers.")
    p.add_argument("--wandb_project", default="fara-sft",
                   help="W&B project name. Used only when 'wandb' is in --report_to.")
    p.add_argument("--wandb_run_name", default=None,
                   help="W&B run name. Defaults to the basename of --output_dir.")
    p.add_argument("--wandb_tags", default=None,
                   help="Comma-separated W&B tags (e.g. 'lora,multiscale').")
    p.add_argument("--wandb_entity", default=None,
                   help="W&B entity (team/user). Falls back to default login if unset.")

    # Toggles
    p.add_argument("--no_multiscale", action="store_true",
                   help="Disable multi-scale patching (single-scale path).")
    p.add_argument("--lora", action="store_true",
                   help="LoRA fine-tune instead of full fine-tune.")
    p.add_argument("--sampling_strategy", default="decision_point",
                    choices=["none", "decision_point", "full_trajectory"],
                   help="How to sample within each trajectory at training time. "
                        "'none' = train on the whole trajectory as-is. "
                        "'decision_point' = Strategy A: sample one random step t "
                        "per row, only train on the action at that step, with "
                        "the last max_n_images_train screenshots kept in context. "
                        "'full_trajectory' = use the final turn of the full trajectory "
                        "with the last max_n_images_train screenshots.")
    p.add_argument("--max_n_images_train", type=int, default=3,
                    help="Image budget when sampling_strategy is 'decision_point' or 'full_trajectory'. "
                        "Mirrors FaraAgent.max_n_images at inference (default 3).")
    p.add_argument("--freeze_vision", action="store_true",
                   help="Freeze the vision tower (full FT only).")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--attn_implementation", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2", "flash_attention_3"],
                   help="Attention kernel. sdpa is safest; FA2/FA3 require the "
                        "model to explicitly advertise support.")
    p.add_argument("--fsdp", default="",
                   help="Comma-separated FSDP options, e.g. 'full_shard,auto_wrap'.")
    p.add_argument("--fsdp_config", default=None,
                   help="Path to FSDP config json file.")

    # LoRA knobs
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj",
                   help="Comma-separated list of module name suffixes.")

    return p.parse_args()


def load_processor_and_model(args: argparse.Namespace):
    processor = FaraProcessor.from_pretrained(args.model_id)
    processor.image_processor = Qwen2VLImageProcessor.from_pretrained(args.model_id)
    processor.image_processor.use_multiscale = not args.no_multiscale

    dtype = torch.bfloat16 if args.bf16 and not args.fp16 else (
        torch.float16 if args.fp16 else torch.float32
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )

    if args.freeze_vision and not args.lora:
        for p in model.visual.parameters():
            p.requires_grad = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return processor, model


def maybe_wrap_peft(model, args):
    if not args.lora:
        return model, None
    from peft import LoraConfig

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    # Let SFTTrainer apply PEFT from `peft_config` to avoid double-wrapping.
    return model, lora_config


def build_dataset(args: argparse.Namespace, processor: Qwen2_5_VLProcessor):
    filter_cache = None
    if args.data_cache_root:
        filter_cache = os.path.join(
            args.data_cache_root, f"filtered-{_filter_cache_key(args)}"
        )

    def _load_and_filter() -> Any:
        if os.path.isdir(args.data_path):
            ds_local = load_dataset("parquet", data_dir=args.data_path, split="train", num_proc=4)
        else:
            ds_local = load_dataset("parquet", data_files=args.data_path, split="train", num_proc=4)

        allowed_domains = [d.strip().lower() for d in args.allowed_domains.split(",") if d.strip()]
        allowed_domains = [d[4:] if d.startswith("www.") else d for d in allowed_domains]
        if not allowed_domains:
            return ds_local

        log(f"[fara-train] applying domain allowlist ({args.domain_filter_mode}): {allowed_domains}")

        def _keep_row(row: Dict[str, Any]) -> bool:
            domains = _extract_row_domains(row.get("trajectory"))
            if not domains:
                return False
            if args.domain_filter_mode == "strict":
                return all(_domain_allowed(d, allowed_domains) for d in domains)
            return any(_domain_allowed(d, allowed_domains) for d in domains)

        before = len(ds_local)
        ds_local = ds_local.filter(_keep_row, num_proc=4)
        after = len(ds_local)
        log(f"[fara-train] domain filter kept {after}/{before} rows")
        if after == 0:
            raise ValueError(
                "Domain filter removed all rows. Verify --allowed_domains and trajectory URL extraction."
            )
        return ds_local

    # Multi-rank coordination: main builds + saves, others wait + load_from_disk.
    # The HF Trainer process group isn't initialized yet at this point in main(),
    # so torch.distributed.barrier() would be a no-op. Use a filesystem sentinel
    # instead — main writes it after save_to_disk completes, others poll for it.
    if filter_cache:
        sentinel = filter_cache + ".done"
        if is_main_process():
            if not os.path.isdir(filter_cache):
                ds = _load_and_filter()
                log(f"[fara-train] saving filtered dataset to {filter_cache}")
                os.makedirs(args.data_cache_root, exist_ok=True)
                ds.save_to_disk(filter_cache)
                del ds  # re-loaded from cache below for state parity with other ranks
            open(sentinel, "w").close()
        else:
            while not os.path.exists(sentinel):
                time.sleep(2)

        log(f"[fara-train] loading filtered dataset from cache: {filter_cache}")
        ds = load_from_disk(filter_cache)
    else:
        # Caching disabled: every rank rebuilds independently. CPU-redundant
        # but no on-disk race.
        ds = _load_and_filter()

    if args.shuffle_seed is not None and args.shuffle_seed >= 0:
        log(f"[fara-train] shuffling dataset with seed={args.shuffle_seed} "
            f"(n={len(ds)})")
        ds = ds.shuffle(seed=args.shuffle_seed)

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # Match Fara agent runtime prompt format as closely as possible.
    sample = ds[0]
    sample_images = sample.get("images", [])
    if len(sample_images) > 0:
        first_image = _decode_image(sample_images[0])
        im_proc = processor.image_processor
        prompt_info = get_computer_use_system_prompt(
            first_image,
            {
                "patch_size": im_proc.patch_size,
                "merge_size": im_proc.merge_size,
                "min_pixels": im_proc.min_pixels,
                "max_pixels": im_proc.max_pixels,
            },
            include_input_text_key_args=False,
            fn_call_template="default",
        )
        system_prompt_text = _extract_system_prompt_text(prompt_info)
    else:
        system_prompt_text = None

    ds = ds.map(
        row_to_messages,
        fn_kwargs={"system_prompt_text": system_prompt_text},
        remove_columns=ds.column_names,
        num_proc=4,
        writer_batch_size=128,
    )

    # Drop trajectories where any step had an unmapped action — `row_to_messages`
    # signals these by returning {"messages": [], "images": []}.
    before = len(ds)
    ds = ds.filter(lambda x: len(x["messages"]) > 0, num_proc=4)
    after = len(ds)
    if before != after:
        log(f"[fara-train] dropped {before - after}/{before} trajectories with unmapped actions; "
              f"{after} remain")
    if after == 0:
        raise ValueError(
            "All trajectories were dropped due to unmapped actions. "
            "Check _format_assistant_message and the dataset's action vocabulary."
        )

    if getattr(args, "viz_dir", None) and is_main_process():
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from visualize_trajectories import render_messages_dataset
        n_viz = min(args.viz_max_rows, len(ds))
        run_name = os.path.basename(os.path.normpath(args.output_dir)) or "run"
        out_path = os.path.join(args.viz_dir, f"{run_name}_build_dataset_sample.html")
        render_messages_dataset(ds.select(range(n_viz)), out_path)
        log(f"[fara-train] wrote post-build viz ({n_viz} rows): {out_path}")

    return ds


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from trl import SFTConfig, SFTTrainer

    log(f"[fara-train] multi-scale = {not args.no_multiscale}")
    log(f"[fara-train] LoRA        = {args.lora}")

    processor, model = load_processor_and_model(args)
    model, lora_config = maybe_wrap_peft(model, args)

    hf_dataset = build_dataset(args, processor)

    eval_hf_dataset = None
    if args.val_split_ratio and args.val_split_ratio > 0:
        if not (0 < args.val_split_ratio < 1):
            raise ValueError(f"--val_split_ratio must be in (0, 1); got {args.val_split_ratio}")
        split = hf_dataset.train_test_split(
            test_size=args.val_split_ratio, seed=args.val_split_seed
        )
        hf_dataset, eval_hf_dataset = split["train"], split["test"]
        if args.val_max_samples is not None:
            eval_hf_dataset = eval_hf_dataset.select(
                range(min(args.val_max_samples, len(eval_hf_dataset)))
            )
        log(f"[fara-train] held-out val set size = {len(eval_hf_dataset)} "
            f"(ratio={args.val_split_ratio}, seed={args.val_split_seed})")

    log(f"[fara-train] train dataset size = {len(hf_dataset)}")

    train_dataset = FaraDataset(
        hf_dataset=hf_dataset,
        processor=processor,
        use_multiscale=not args.no_multiscale,
        sampling_strategy=args.sampling_strategy,
        max_n_images=args.max_n_images_train,
        max_seq_length=args.max_seq_length,
    )

    eval_dataset = None
    if eval_hf_dataset is not None:
        eval_dataset = FaraDataset(
            hf_dataset=eval_hf_dataset,
            processor=processor,
            use_multiscale=not args.no_multiscale,
            sampling_strategy=args.sampling_strategy,
            max_n_images=args.max_n_images_train,
            max_seq_length=args.max_seq_length,
        )

    tok = processor.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    collator = FaraCollator(pad_id=pad_id)

    fsdp_options: List[str] = []
    if args.fsdp:
        fsdp_options = [opt.strip() for opt in args.fsdp.split(",") if opt.strip()]

    # Avoid passing None: some TRL/Transformers versions do membership checks on fsdp.
    fsdp_value = fsdp_options if fsdp_options else ""

    report_to = [x.strip() for x in args.report_to.split(",") if x.strip()]
    if len(report_to) == 0 or report_to == ["none"]:
        report_to = []

    # W&B env vars are picked up by Transformers' WandbCallback at init time.
    if "wandb" in report_to:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_tags:
            os.environ["WANDB_TAGS"] = args.wandb_tags
    run_name = args.wandb_run_name or os.path.basename(args.output_dir.rstrip("/"))

    log(f"[fara-train] training config: {json.dumps(vars(args), indent=2, default=str)}")

    train_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size or args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        bf16=args.bf16 and not args.fp16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=args.max_seq_length,
        report_to=report_to,
        run_name=run_name,
        save_total_limit=args.save_total_limit if args.save_total_limit else None,
        optim="adamw_torch",
        fsdp=fsdp_value,
        fsdp_config=args.fsdp_config,
    )

    # Resolve vision LR: explicit > ratio.
    if args.vision_learning_rate is not None:
        train_args.vision_learning_rate = args.vision_learning_rate
    else:
        train_args.vision_learning_rate = args.learning_rate * args.vision_lr_ratio
    log(f"[fara-train] LLM lr={args.learning_rate:.2e} | "
        f"vision lr={train_args.vision_learning_rate:.2e}")

    class PatchStatsTrainer(PatchStatsSFTTrainer, SFTTrainer):
        pass

    trainer = PatchStatsTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
        peft_config=lora_config,
    )
    trainer._init_patch_stats()

    if (
        eval_hf_dataset is not None
        and "wandb" in report_to
        and args.inf_eval_samples > 0
    ):
        trainer.add_callback(
            InferenceEvalCallback(
                eval_hf_dataset=eval_hf_dataset,
                processor=processor,
                n_samples=args.inf_eval_samples,
                max_n_images=args.max_n_images_train,
                max_new_tokens=args.inf_eval_max_new_tokens,
                seed=args.inf_eval_seed,
            )
        )
        log(f"[fara-train] inference eval callback: {args.inf_eval_samples} "
            f"samples per eval (seed={args.inf_eval_seed})")

    trainer.train()

    if getattr(trainer, "_patch_samples", 0) > 0 and getattr(trainer, "_patch_baseline_tokens", 0) > 0:
        saved_tokens = trainer._patch_baseline_tokens - trainer._patch_final_tokens
        reduction_pct = (saved_tokens / trainer._patch_baseline_tokens) * 100.0
        log(
            f"[fara-train] training complete: saved {saved_tokens} tokens "
            f"({reduction_pct:.1f}% reduction) over {trainer._patch_samples} samples"
        )

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
