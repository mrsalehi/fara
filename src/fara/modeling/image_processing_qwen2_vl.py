# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for Qwen2-VL."""

import math
from typing import Optional, Union

import numpy as np

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType, logging
from transformers.video_utils import VideoInput, make_batched_videos

from .trajectory_patch import process_trajectory


logger = logging.get_logger(__name__)


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Qwen2VLImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Qwen2-VL image processor that dynamically resizes images based on the original images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        size (`dict[str, int]`, *optional*, defaults to `{"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}`):
            Size of the image after resizing. `shortest_edge` and `longest_edge` keys must be present.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `list[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `list[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel in the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_pixels (`int`, *optional*, defaults to `56 * 56`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `28 * 28 * 1280`):
            The max pixels of the image to resize the image.
        patch_size (`int`, *optional*, defaults to 14):
            The spatial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
    """

    model_input_names = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        else:
            size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}
        # backward compatibility: override size with min_pixels and max_pixels if they are provided
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
        self.min_pixels = size["shortest_edge"]
        self.max_pixels = size["longest_edge"]
        self.size = size

        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD

        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.do_convert_rgb = do_convert_rgb
        self.scales = [0, 1, 2, 3]
        self.use_multiscale = True

    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        traj_per_frame=None,
    ):
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.
            vision_info (`list[Dict]`, *optional*):
                Optional list of dictionaries containing additional information about vision inputs.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. `shortest_edge` and `longest_edge` keys must be present.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the `PILImageResampling` enums.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Scale factor to use if rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                The spatial patch size of the vision encoder.
            temporal_patch_size (`int`, *optional*, defaults to `self.temporal_patch_size`):
                The temporal patch size of the vision encoder.
            merge_size (`int`, *optional*, defaults to `self.merge_size`):
                The merge size of the vision encoder to llm encoder.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.   - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []

        assert len(images) == 1, "Fara agent only supports one image input for now, but got {} images.".format(len(images))

        # All resizes happen on uint8 (no PIL precision round-trip). Rescale and
        # normalize are applied per-scale after the resize+pad below, and once on
        # the base image after the loop.
        patches_multi_scale = []
        positions_multi_scale = []
        centers_multi_scale = []
        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                image = resize(
                    image, size=(resized_height, resized_width), resample=resample, input_data_format=input_data_format
                )

            if traj_per_frame is not None:
                # resize the image to multiple scales
                for s, coords in traj_per_frame[0]['kept_by_size'].items():
                    resize_scale = (s // (patch_size * merge_size))
                    resized_image_this_scale = resize(
                        image,
                        size=(resized_height // resize_scale, resized_width // resize_scale),
                        resample=resample,
                        input_data_format=input_data_format,
                    )
                    # pad resized_image_this_scale so the width and height are divisible by patch_size * merge_size
                    merge_patch_size = patch_size * merge_size
                    pad_h = -resized_image_this_scale.shape[0] % merge_patch_size
                    pad_w = -resized_image_this_scale.shape[1] % merge_patch_size
                    resized_image_this_scale = np.pad(
                        resized_image_this_scale,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode='edge',
                    )

                    if do_rescale:
                        resized_image_this_scale = self.rescale(
                            resized_image_this_scale,
                            scale=rescale_factor,
                            input_data_format=input_data_format,
                        )

                    if do_normalize:
                        resized_image_this_scale = self.normalize(
                            image=resized_image_this_scale,
                            mean=image_mean,
                            std=image_std,
                            input_data_format=input_data_format,
                        )

                    N = coords.shape[0]
                    patches_this_scale = np.stack([
                        resized_image_this_scale[
                            y//resize_scale:y//resize_scale + merge_patch_size,
                            x//resize_scale:x//resize_scale + merge_patch_size
                            ]
                        for x, y in coords
                    ])

                    patches_this_scale = patches_this_scale.reshape(N, merge_size, patch_size, merge_size, patch_size, 3).transpose(0, 1, 3, 2, 4, 5).reshape(N*4, patch_size, patch_size, 3)
                    if data_format == ChannelDimension.FIRST:
                        patches_this_scale = patches_this_scale.transpose(0, 3, 1, 2)

                    # span = s // merge_patch_size
                    # positions_this_scale = np.array([[y // merge_patch_size + (span-1)/2, x // merge_patch_size + (span-1)/2] for x, y in coords])
                    half = s // 2
                    sub_offsets = [
                        (0,    0   ),   # TL  (dy, dx)
                        (0,    half),   # TR
                        (half, 0   ),   # BL
                        (half, half),   # BR
                    ]

                    positions_this_scale = np.array([
                        [(y + dy) // self.patch_size, (x + dx) // self.patch_size]
                        for x, y in coords
                        for dy, dx in sub_offsets
                    ], dtype=np.int64)
                
                    
                    # window attention book keeping
                    centers_this_scale = np.array([
                        [x+s/2, y+s/2] for y, x in coords
                    ], dtype=np.int64)
                    centers_multi_scale.append(centers_this_scale)


                    positions_multi_scale.append(positions_this_scale)
                    patches_multi_scale.append(patches_this_scale)

            # The base image is only consumed in the no-trajectory path (it gets
            # appended to `processed_images` and reshaped into a uniform patch grid
            # downstream). In the multi-scale path, downstream code reads from
            # `patches_multi_scale` instead, so skip the work here.
            if traj_per_frame is None:
                if do_rescale:
                    image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)

                if do_normalize:
                    image = self.normalize(
                        image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                    )

                image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
                processed_images.append(image)

        if traj_per_frame is None:
            patches = np.array(processed_images)
        else:
            patches = np.concatenate(patches_multi_scale, axis=0)[None, ...]  # shape: (1, num_patches, patch_size, patch_size, 3)
            positions_multi_scale = np.concatenate(positions_multi_scale, axis=0)
            centers_multi_scale = np.concatenate(centers_multi_scale, axis=0)

        if data_format == ChannelDimension.LAST:
            if patches.ndim == 4:
                patches = patches.transpose(0, 3, 1, 2)
            elif patches.ndim == 5:
                patches = patches.transpose(0, 1, 4, 2, 3)

        if patches.shape[0] % temporal_patch_size != 0:
            repeats = np.repeat(
                patches[-1][np.newaxis], temporal_patch_size - (patches.shape[0] % temporal_patch_size), axis=0
            )
            patches = np.concatenate([patches, repeats], axis=0)

        if traj_per_frame is None:
            channel = patches.shape[1]
            grid_t = patches.shape[0] // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
            patches = patches.reshape(
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
            flatten_patches = patches.reshape(
                grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
            )
        else:
            num_patches = patches.shape[1]
            grid_t = patches.shape[0] // temporal_patch_size
            # lying about the grid shape as it doesn't matter for the multi-scale
            # the only thing that matters is that grid_t*grid_h*grid_w == num_patches after flattening
            grid_h = 2
            grid_w = num_patches // 2
            flatten_patches = patches.transpose(1, 2, 0, 3, 4).reshape(-1, temporal_patch_size*patch_size*patch_size*3)
            # print(f"Image processor: flatten_patches shape: {flatten_patches.shape}, grid_t: {grid_t}, grid_h: {grid_h}, grid_w: {grid_w}, num_patches: {num_patches}")
            return flatten_patches, (grid_t, grid_h, grid_w), positions_multi_scale, centers_multi_scale

        return flatten_patches, (grid_t, grid_h, grid_w), None, None

    def _compute_trajectory_patches(
        self,
        images,
        do_convert_rgb,
        do_resize,
        size,
        resample,
        patch_size,
        merge_size,
        input_data_format,
        var_thresh=100.0,
        mse_thresh=10.0,
        prev_traj_data=None,
    ):
        """
        Run the trajectory patch algorithm on the full frame list.

        Returns a dict:
          {
            'frames': list[np.ndarray]  # HWC uint8, post-resize frames fed into the algo
            'per_frame': list[{
                'frame_idx': int,
                'kept_patches':    list[(x, y, size)],
                'dropped_patches': list[(x, y, size)],  # redundant / unchanged regions
                'img_size':        (w, h),
                'scroll_dy':       int,
            }],
            'n_skipped_blank':      int,
            'trailing_dup_dropped': bool,
          }
        or None if `images` is empty.
        """
        from fara.modeling.trajectory_patch import process_trajectory
        from fara.modeling.trajectory_patch import group_patches_by_size as _group_by_size

        if not images:
            return None

        def _resize_one(img):
            """Run the same convert/resize/format pipeline used for full trajectories
            on a single image. Returns an HWC uint8 numpy array.
            """
            f = img
            if do_convert_rgb:
                f = convert_to_rgb(f)
            f = to_numpy_array(f)
            fmt_local = input_data_format or infer_channel_dimension_format(f)
            if do_resize:
                h, w = get_image_size(f, channel_dim=fmt_local)
                rh, rw = smart_resize(
                    h, w,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                f = resize(f, size=(rh, rw), resample=resample, input_data_format=fmt_local)
            f = to_channel_dimension_format(f, ChannelDimension.LAST, input_channel_dim=fmt_local)
            if f.dtype != np.uint8:
                f = np.clip(f, 0, 255).astype(np.uint8)
            return f

        def _build_per_frame(r):
            return {
                'frame_idx':       r['frame_idx'],
                'kept_patches':    r['kept_patches'],
                'dropped_patches': r['dropped_patches'],
                'kept_by_size':    _group_by_size(r['kept_patches']),
                'dropped_by_size': _group_by_size(r['dropped_patches']),
                'img_size':        r['img_size'],
                'scroll_dy':       r['scroll_dy'],
                '_img':            r['img'],
            }

        # Fast path: caller passed cached traj_data for an N-1 length history and
        # appended exactly one new image. Reuse cached frames/per_frame for indices
        # 0..N-2 and only diff the new frame against the previous one.
        if (
            prev_traj_data is not None
            and len(prev_traj_data.get('frames', [])) == len(images) - 1
            and len(images) >= 2
        ):
            new_frame = _resize_one(images[-1])
            prev_frame = prev_traj_data['frames'][-1]
            # 2-frame call: frame 0 result is intra-frame quadtree on prev (discarded);
            # frame 1 is the diff of new_frame against prev_frame (the only one we keep).
            pair_results = process_trajectory(
                [prev_frame, new_frame], var_thresh=var_thresh, mse_thresh=mse_thresh,
            )
            new_entry = _build_per_frame(pair_results[1])
            new_entry['frame_idx'] = len(images) - 1
            return {
                'frames':    list(prev_traj_data['frames']) + [new_frame],
                'per_frame': list(prev_traj_data['per_frame']) + [new_entry],
            }

        # Slow path: full recomputation across the whole image list.
        traj_frames = [_resize_one(f) for f in images]
        traj_results = process_trajectory(
            traj_frames, var_thresh=var_thresh, mse_thresh=mse_thresh,
        )
        per_frame = [_build_per_frame(r) for r in traj_results]
        return {
            'frames': traj_frames,
            'per_frame': per_frame,
        }

    def _compute_image_status(self, traj_data, images) -> list:
        """Per-input-image status used to keep messages and pixel_values aligned.

        Returns 'active' or 'mid_dup' per input image. 'mid_dup' is flagged
        when multiscale process_trajectory produced an empty kept_patches list
        (i.e. the frame is fully redundant relative to its predecessor).

        Blank-prefix and trailing-duplicate frames are handled offline by
        scripts/preprocess_trajectories.py and never reach this codepath at
        training time. Without traj_data (single-scale), every frame is active.
        """
        if traj_data is None:
            return ['active'] * len(images)
        per_frame = traj_data['per_frame']
        return [
            'mid_dup' if len(per_frame[i].get('kept_patches', [])) == 0 else 'active'
            for i in range(len(per_frame))
        ]

    def _dump_empty_kept_case(self, idx, current_image, images, traj_data):
        """Dump the current frame + its predecessor + diff_frame visualization
        when kept_patches is empty, so we can see what caused the failure.
        Output dir: $FARA_EMPTY_KEPT_DUMP_DIR (default /tmp/fara_empty_kept).
        """
        import os, time
        from PIL import Image as _PIL
        from fara.modeling.trajectory_patch import visualize_frame

        dump_dir = os.environ.get("FARA_EMPTY_KEPT_DUMP_DIR", "/tmp/fara_empty_kept")
        os.makedirs(dump_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(dump_dir, f"empty_kept_{ts}_pid{os.getpid()}_idx{idx}")
        os.makedirs(run_dir, exist_ok=True)

        def _to_pil(x):
            if isinstance(x, _PIL.Image):
                return x.convert("RGB")
            arr = np.asarray(x)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = arr.transpose(1, 2, 0)
            return _PIL.fromarray(arr)

        try:
            _to_pil(current_image).save(os.path.join(run_dir, "curr.png"))
        except Exception as e:
            print(f"[fara] failed to save curr.png: {e}")
        if idx > 0:
            try:
                _to_pil(images[idx - 1]).save(os.path.join(run_dir, "prev.png"))
            except Exception as e:
                print(f"[fara] failed to save prev.png: {e}")

        info = traj_data['per_frame'][idx]
        try:
            visualize_frame(
                info['_img'], info['kept_patches'], info['dropped_patches'],
                info['frame_idx'], info['scroll_dy'],
                os.path.join(run_dir, "viz_kept_dropped.png"),
            )
        except Exception as e:
            print(f"[fara] visualize_frame failed: {e}")

        import json as _json
        with open(os.path.join(run_dir, "info.txt"), "w") as f:
            f.write(f"idx={idx}\n")
            f.write(f"frame_idx={info.get('frame_idx')}\n")
            f.write(f"scroll_dy={info.get('scroll_dy')}\n")
            f.write(f"img_size={info.get('img_size')}\n")
            f.write(f"kept_patches={len(info.get('kept_patches', []))}\n")
            f.write(f"dropped_patches={len(info.get('dropped_patches', []))}\n")
            f.write(f"n_total_frames={len(traj_data['per_frame'])}\n")

        # Dump collator-level context (chat text + messages) if the collator
        # set self._collator_debug_info before calling preprocess.
        debug = getattr(self, "_collator_debug_info", None)
        if debug:
            if "text" in debug:
                with open(os.path.join(run_dir, "chat_text.txt"), "w") as f:
                    f.write(debug["text"])
            if "messages" in debug:
                with open(os.path.join(run_dir, "messages.json"), "w") as f:
                    _json.dump(debug["messages"], f, indent=2, default=str)

        print(f"[fara] empty kept_patches at idx={idx}; dumped to {run_dir}", flush=True)

    def _maybe_visualize_trajectory(self, traj_data):
        import os
        viz_dir = os.environ.get("FARA_TRAJ_VIZ_DIR")
        if not viz_dir or traj_data is None:
            return

        from fara.modeling.trajectory_patch import visualize_frame
        import hashlib

        frames = traj_data['frames']
        task_id = os.environ.get("FARA_TRAJ_TASK_ID")
        if not task_id:
            h = hashlib.md5()
            h.update(frames[0].tobytes())
            h.update(str(len(frames)).encode())
            task_id = h.hexdigest()[:12]
        run_dir = os.path.join(viz_dir, f"traj_{task_id}")
        for r in traj_data['per_frame']:
            out_path = os.path.join(run_dir, f"frame_{r['frame_idx']:03d}.png")
            visualize_frame(
                r['_img'], r['kept_patches'], r['dropped_patches'],
                r['frame_idx'], r['scroll_dy'], out_path,
            )

    def preprocess(
        self,
        images: ImageInput,
        videos: VideoInput = None,
        do_resize: Optional[bool] = None,
        size: Optional[dict[str, int]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        prev_traj_data: Optional[dict] = None,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            videos (`VideoInput`):
                Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If
                passing in videos with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `list[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `list[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            min_pixels (`int`, *optional*, defaults to `self.min_pixels`):
                The min pixels of the image to resize the image.
            max_pixels (`int`, *optional*, defaults to `self.max_pixels`):
                The max pixels of the image to resize the image.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                The spatial patch size of the vision encoder.
            temporal_patch_size (`int`, *optional*, defaults to `self.temporal_patch_size`):
                The temporal patch size of the vision encoder.
            merge_size (`int`, *optional*, defaults to `self.merge_size`):
                The merge size of the vision encoder to llm encoder.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        """
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels

        if size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
            min_pixels = size["shortest_edge"]
        elif min_pixels is not None and max_pixels is not None:
            # backward compatibility: override size with min_pixels and max_pixels if they are provided
            size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        else:
            size = {**self.size}

        do_resize = do_resize if do_resize is not None else self.do_resize

        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if images is not None:
            images = make_flat_list_of_images(images)

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        validate_preprocess_arguments(
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        data = {}
        if images is not None:
            if self.use_multiscale:
                traj_data = self._compute_trajectory_patches(
                    images,
                    do_convert_rgb=do_convert_rgb,
                    do_resize=do_resize,
                    size=size,
                    resample=resample,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    input_data_format=input_data_format,
                    prev_traj_data=prev_traj_data,
                )
            else:
                traj_data = None
            self._maybe_visualize_trajectory(traj_data)

            # Decide a per-input-image status the parent processor can use to
            # update messages in sync with the filtered image set:
            #   "active"   — visual content kept, included in pixel_values/grid_thw
            #   "mid_dup"  — fully redundant frame (kept=[]); drop image, replace text
            # Blank-prefix and trailing-dup frames are handled offline.
            image_status: list = self._compute_image_status(traj_data, images)

            pixel_values, vision_grid_thws = [], []
            all_positions, all_centers = [], []
            active_pf_idx = 0
            for idx, image in enumerate(images):
                if image_status[idx] != 'active':
                    continue
                per_frame = (
                    [traj_data['per_frame'][idx]] if traj_data is not None else None
                )
                active_pf_idx += 1
                patches, image_grid_thw, pos_ms, ctr_ms = self._preprocess(
                    image,
                    do_resize=do_resize,
                    size=size,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=merge_size,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                    traj_per_frame=per_frame
                )
                pixel_values.extend(patches)
                vision_grid_thws.append(image_grid_thw)
                if pos_ms is not None:
                    all_positions.append(pos_ms)
                    all_centers.append(ctr_ms)

            # data["image_status"] = image_status
            if pixel_values:
                data["pixel_values"]   = np.array(pixel_values)
                data["image_grid_thw"] = np.array(vision_grid_thws)
            if all_positions:
                data["maybe_positions_multiscale"] = np.concatenate(all_positions, axis=0)
            if all_centers:
                data["maybe_centers_multiscale"] = np.concatenate(all_centers, axis=0)

        # Extract image_status before creating BatchFeature (strings can't be tensorized)
        # image_status_list = data.pop("image_status", None)

        # kept for BC only and should be removed after v5.0
        if videos is not None:
            logger.warning(
                "`Qwen2VLImageProcessor` works only with image inputs and doesn't process videos anymore. "
                "This is a deprecated behavior and will be removed in v5.0. "
                "Your videos should be forwarded to `Qwen2VLVideoProcessor`. "
            )
            videos = make_batched_videos(videos)
            pixel_values_videos, vision_grid_thws_videos = [], []
            for images in videos:
                patches, video_grid_thw = self._preprocess(
                    images,
                    do_resize=do_resize,
                    size=size,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=merge_size,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                pixel_values_videos.extend(patches)
                vision_grid_thws_videos.append(video_grid_thw)
            data.update(
                {
                    "pixel_values_videos": np.array(pixel_values_videos),
                    "video_grid_thw": np.array(vision_grid_thws_videos),
                }
            )

        result = BatchFeature(data=data, tensor_type=return_tensors)

        # Add image_status back (not tensorized, stored as-is)
        if image_status is not None:
            result["image_status"] = image_status
        # Surface traj_data so callers (FaraAgent at inference) can re-cache it
        # and pass it back as `prev_traj_data` next step to avoid recomputing
        # diffs for older frames. Stored as a Python dict (not tensorized).
        if images is not None and self.use_multiscale and traj_data is not None:
            result["traj_data"] = traj_data

        return result

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs=None):
        """
        A utility that returns number of image patches for a given image size.

        Args:
            height (`int`):
                Height of the input image.
            width (`int`):
                Width of the input image.
            images_kwargs (`dict`, *optional*)
                Any kwargs to override defaults of the image processor.
        Returns:
            `int`: Number of image patches per image.
        """
        min_pixels = images_kwargs.get("min_pixels", None) or self.size["shortest_edge"]
        max_pixels = images_kwargs.get("max_pixels", None) or self.size["longest_edge"]
        patch_size = images_kwargs.get("patch_size", None) or self.patch_size
        merge_size = images_kwargs.get("merge_size", None) or self.merge_size

        factor = patch_size * merge_size
        resized_height, resized_width = smart_resize(
            height, width, factor, min_pixels=min_pixels, max_pixels=max_pixels
        )
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        return grid_h * grid_w


__all__ = ["Qwen2VLImageProcessor"]
