"""FaraProcessor: Qwen2.5-VL processor extended with image filtering.

Accepts raw `messages` + PIL `images` directly, runs blank/duplicate
filtering on the image sequence, then applies the chat template and
standard tokenization/image processing — all in one call.

This makes the filtering available at both training time (FaraDataset)
and inference time (FaraAgent), with no duplication.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLProcessor
from transformers.feature_extraction_utils import BatchFeature

# Production strings that define the user-message format.
USER_MESSAGE = "Here is the next screenshot. Think about what to do next."
USER_MESSAGE_REDUNDANT = (
    "The screenshot is unchanged from the previous step. Think about what to do next."
)


def _to_array(img: Any) -> np.ndarray:
    """Convert any image representation to a uint8 HWC numpy array."""
    if isinstance(img, np.ndarray):
        return img.astype(np.uint8) if img.dtype != np.uint8 else img
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
    if isinstance(img, (bytes, bytearray)):
        return np.asarray(Image.open(io.BytesIO(bytes(img))).convert("RGB"), dtype=np.uint8)
    if isinstance(img, dict) and "bytes" in img and img["bytes"] is not None:
        return np.asarray(Image.open(io.BytesIO(img["bytes"])).convert("RGB"), dtype=np.uint8)
    raise TypeError(f"Cannot convert to array: {type(img)}")


def _to_pil(img: Any) -> Image.Image:
    """Convert any image representation to a PIL RGB image."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        arr = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = arr.transpose(1, 2, 0)
        return Image.fromarray(arr)
    if isinstance(img, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(img))).convert("RGB")
    if isinstance(img, dict) and "bytes" in img and img["bytes"] is not None:
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    raise TypeError(f"Cannot convert to PIL: {type(img)}")


class FaraProcessor(Qwen2_5_VLProcessor):
    """Qwen2.5-VL processor with image-sequence filtering built in.

    The key addition: accepts raw `messages` (a structured list of chat
    turns) in addition to / instead of pre-templated `text`.  When
    `messages` is provided, `_filter_images` is called first to drop
    blank leading frames, trailing duplicates, and mid-trajectory
    identical frames before the chat template is applied.  This keeps
    the filtering logic in one place and makes it available at inference
    as well as training.
    """

    def __call__(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        images: Optional[List[Any]] = None,
        text: Optional[Union[str, List[str]]] = None,
        videos=None,
        **kwargs,
    ) -> BatchFeature:
        if messages is not None:
            pil_images = [_to_pil(img) for img in images] if images else []

            messages, pil_images = self._filter_images(messages, pil_images)

            text = self.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            if not isinstance(text, list):
                text = [text]
            images = pil_images or None

        return super().__call__(
            images=images, text=text, videos=videos, **kwargs
        )

    def _filter_images(
        self,
        messages: List[Dict[str, Any]],
        images: List[Image.Image],
    ) -> tuple:
        """Filter the image sequence and keep messages in sync.

        Three passes:
        1. Trailing duplicate  — drop last image + last (user, assistant) pair if the
           image is a duplicate of the penultimate image.
        2. Leading blank prefix — frames with pixel variance == 0 (about:blank).
           user_0 keeps its text but loses the image placeholder; any later
           pre-render turns (user_1 .. user_{n-1}) are dropped entirely together
           with their assistant turns.
        3. Mid-trajectory duplicates — pixel-identical consecutive frames.
           The image is dropped; the user-turn text is replaced with
           USER_MESSAGE_REDUNDANT so the model knows the page didn't change.
        """
        if not images:
            return messages, images

        arrays = [_to_array(img) for img in images]

        def _blank(a):
            return float(a.astype(np.float32).var()) == 0.0

        def _eq(a, b):
            return a.shape == b.shape and np.array_equal(a, b)

        # ── 1. Trailing duplicate ─────────────────────────────────────────────
        if len(arrays) >= 2 and _eq(arrays[-1], arrays[-2]):
            arrays = arrays[:-1]
            images = images[:-1]
            # Remove last user turn and its following assistant turn.
            for mi in range(len(messages) - 1, -1, -1):
                if messages[mi]["role"] == "user":
                    messages = messages[:mi] + messages[mi + 2:]
                    break

        # ── 2. Leading blank count ────────────────────────────────────────────
        n_blank = 0
        while n_blank < len(arrays) and _blank(arrays[n_blank]):
            n_blank += 1

        # ── 3. Mid-trajectory dup flags ───────────────────────────────────────
        is_mid_dup = [False] * len(arrays)
        for i in range(1, len(arrays)):
            if i >= n_blank and _eq(arrays[i], arrays[i - 1]):
                is_mid_dup[i] = True

        if n_blank == 0 and not any(is_mid_dup):
            return messages, images   # nothing to do

        # ── 4. Single-pass rebuild ────────────────────────────────────────────
        new_messages: List[Dict[str, Any]] = []
        new_images: List[Image.Image] = []
        ci = 0
        skip_next = False   # set True to drop the assistant turn after a dropped user

        for msg in messages:
            if skip_next:
                skip_next = False
                continue

            if msg["role"] != "user":
                new_messages.append(msg)
                continue

            has_img = any(
                isinstance(c, dict) and c.get("type") == "image"
                for c in msg["content"]
            )
            if not has_img:
                new_messages.append(msg)
                continue

            current_ci = ci
            ci += 1

            if 0 < current_ci < n_blank:
                # Leading blank, i >= 1: drop user turn AND following assistant.
                skip_next = True
                continue

            if current_ci < n_blank:
                # Leading blank, i == 0: keep task text, strip image placeholder.
                content = [
                    c for c in msg["content"]
                    if not (isinstance(c, dict) and c.get("type") == "image")
                ]
                new_messages.append({"role": "user", "content": content})
            elif is_mid_dup[current_ci]:
                # Mid-traj dup: drop image, swap to USER_MESSAGE_REDUNDANT marker.
                content = []
                for c in msg["content"]:
                    if isinstance(c, dict) and c.get("type") == "image":
                        continue
                    if isinstance(c, dict) and c.get("type") == "text":
                        content.append({
                            "type": "text",
                            "text": c["text"].replace(USER_MESSAGE, USER_MESSAGE_REDUNDANT),
                        })
                    else:
                        content.append(c)
                new_messages.append({"role": "user", "content": content})
            else:
                new_messages.append(msg)
                new_images.append(images[current_ci])

        return new_messages, new_images
