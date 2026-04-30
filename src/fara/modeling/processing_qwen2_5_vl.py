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
from transformers.image_utils import ImageInput
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessorKwargs

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
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos=None,
        messages: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
            - **second_per_grid_ts** -- List of video seconds per time grid. Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = videos_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            # The image processor returns a per-input-image status list to keep
            # message placeholders aligned with the (filtered) pixel tensors.
            image_status = image_inputs.pop("image_status", None)
            # When messages were provided, update them in sync with image_status,
            # then derive the templated text from the filtered messages.
            if messages is not None and image_status is not None:
                messages = self._apply_image_status(messages, image_status)
                text = self.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
            image_grid_thw = image_inputs.get("image_grid_thw")
        elif messages is not None:
            text = self.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )

        if videos is not None:
            fps = output_kwargs["videos_kwargs"].get("fps", 2.0)
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]

            if isinstance(fps, (int, float)):
                second_per_grid_ts = [self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [self.video_processor.temporal_patch_size / tmp for tmp in fps]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        if images is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    num_video_tokens = video_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.video_token, "<|placeholder|>" * num_video_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

    def _apply_image_status(
        self,
        messages: List[Dict[str, Any]],
        image_status: List[str],
    ) -> List[Dict[str, Any]]:
        """Update `messages` in sync with the per-input-image status returned
        by `Qwen2VLImageProcessor.preprocess`. The status drives:
          'active'        → keep message as-is (image stays in pixel_values)
          'blank_first'   → strip image placeholder, keep task text
          'blank_drop'    → drop the user turn AND its following assistant turn
          'trailing_dup'  → drop the user turn AND its following assistant turn
          'mid_dup'       → strip image placeholder, replace USER_MESSAGE
                            with USER_MESSAGE_REDUNDANT marker text
        """
        new_messages: List[Dict[str, Any]] = []
        ci = 0
        skip_next = False

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

            status = image_status[ci] if ci < len(image_status) else "active"
            ci += 1

            if status in ("blank_drop", "trailing_dup"):
                skip_next = True   # also drop the following assistant turn
                continue

            if status == "blank_first":
                content = [
                    c for c in msg["content"]
                    if not (isinstance(c, dict) and c.get("type") == "image")
                ]
                new_messages.append({"role": "user", "content": content})
            elif status == "mid_dup":
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
            else:   # 'active'
                new_messages.append(msg)

        return new_messages
