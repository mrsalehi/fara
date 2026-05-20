#!/usr/bin/env python
"""Post-training inference eval: load each saved checkpoint on a single
GPU (no FSDP), reuse InferenceEvalCallback's rendering path on the same
val split that training used, and log per-checkpoint HTML to wandb under
`eval/inf/samples` at the step that produced the checkpoint.

Mirrors the data-build flags from train_fara.py so the val split is
reconstructed identically (same --val_split_ratio + --val_split_seed
+ --val_max_samples).
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import re
import sys
from types import SimpleNamespace
from typing import List

import torch

# Make scripts/ importable so we can grab the helpers from train_fara.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_fara import (  # noqa: E402
    InferenceEvalCallback,
    build_dataset,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Where the checkpoints live (training --output_dir).
    p.add_argument("--ckpt_dir", required=True,
                   help="Training output dir containing checkpoint-* subdirs.")
    p.add_argument("--ckpt_pattern", default="checkpoint-*")
    p.add_argument("--base_model_id", default="microsoft/Fara-7B",
                   help="Used for the processor and (for LoRA ckpts) the base weights.")
    p.add_argument("--include_base_model", action="store_true",
                   help="Run inference eval on --base_model_id at step=0 before "
                        "iterating saved checkpoints (baseline comparison).")
    p.add_argument("--gpu_id", type=int, default=0)

    # Data-build flags — must match training to reproduce the val split.
    p.add_argument("--data_path", required=True)
    p.add_argument("--data_cache_root", default=None)
    p.add_argument("--allowed_domains", default="")
    p.add_argument("--domain_filter_mode", default="strict",
                   choices=["strict", "any"])
    p.add_argument("--sampling_strategy", default="full_trajectory",
                   choices=["full_trajectory", "decision_point"])
    p.add_argument("--max_n_images_train", type=int, default=3)
    p.add_argument("--max_seq_length", type=int, default=16384)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--shuffle_seed", type=int, default=-1)
    p.add_argument("--no_multiscale", action="store_true")

    p.add_argument("--val_split_ratio", type=float, required=True)
    p.add_argument("--val_split_seed", type=int, default=42)
    p.add_argument("--val_max_samples", type=int, default=None)

    # Model / runtime.
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--attn_implementation", default="sdpa")

    # Inference-eval knobs.
    p.add_argument("--inf_eval_samples", type=int, default=4)
    p.add_argument("--inf_eval_seed", type=int, default=0)
    p.add_argument("--inf_eval_max_new_tokens", type=int, default=256)

    # W&B.
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_project", default="fara-multiscale")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_run_id", default=None,
                   help="If set, resume this run (per-checkpoint HTMLs land "
                        "on the original training timeline).")
    p.add_argument("--wandb_tags", default="")
    return p.parse_args()


def find_checkpoints(ckpt_dir: str, pattern: str) -> List[str]:
    paths = glob.glob(os.path.join(ckpt_dir, pattern))
    paths = [p for p in paths if os.path.isdir(p)]
    step_re = re.compile(r"checkpoint-(\d+)$")
    def _step(p: str) -> int:
        m = step_re.search(os.path.basename(p.rstrip("/")))
        return int(m.group(1)) if m else -1
    paths.sort(key=_step)
    return paths


def step_from_ckpt(path: str) -> int:
    m = re.search(r"checkpoint-(\d+)$", os.path.basename(path.rstrip("/")))
    if not m:
        raise ValueError(f"Could not parse step from {path}")
    return int(m.group(1))


def load_processor(base_model_id: str, use_multiscale: bool):
    # Import lazily so train_fara's heavyweight modeling imports don't fire
    # before we even parsed args.
    from fara.modeling.processing_qwen2_5_vl import FaraProcessor
    from fara.modeling.image_processing_qwen2_vl import Qwen2VLImageProcessor

    processor = FaraProcessor.from_pretrained(base_model_id)
    processor.image_processor = Qwen2VLImageProcessor.from_pretrained(base_model_id)
    processor.image_processor.use_multiscale = use_multiscale
    return processor


def load_model(ckpt_dir: str, base_model_id: str, dtype: torch.dtype,
               attn_implementation: str, device: str):
    from fara.modeling.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

    is_lora = os.path.exists(os.path.join(ckpt_dir, "adapter_config.json"))
    if is_lora:
        from peft import PeftModel
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_id, torch_dtype=dtype, attn_implementation=attn_implementation,
        )
        model = PeftModel.from_pretrained(base, ckpt_dir)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            ckpt_dir, torch_dtype=dtype, attn_implementation=attn_implementation,
        )
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if args.val_split_ratio <= 0:
        sys.exit("--val_split_ratio must be > 0 to construct a val set")

    device = f"cuda:{args.gpu_id}"
    dtype = torch.bfloat16 if (args.bf16 and not args.fp16) else (
        torch.float16 if args.fp16 else torch.float32)

    processor = load_processor(args.base_model_id, not args.no_multiscale)

    # build_dataset reads these attrs on `args`; the parser above provides them.
    hf_dataset = build_dataset(args, processor)
    split = hf_dataset.train_test_split(
        test_size=args.val_split_ratio, seed=args.val_split_seed,
    )
    eval_hf_dataset = split["test"]
    if args.val_max_samples is not None:
        eval_hf_dataset = eval_hf_dataset.select(
            range(min(args.val_max_samples, len(eval_hf_dataset)))
        )
    print(f"[inf-eval] val set size = {len(eval_hf_dataset)}")

    checkpoints = find_checkpoints(args.ckpt_dir, args.ckpt_pattern)
    if not checkpoints:
        sys.exit(f"No checkpoints matching {args.ckpt_pattern} in {args.ckpt_dir}")
    print(f"[inf-eval] found {len(checkpoints)} checkpoints:")
    for p in checkpoints:
        print(f"  {os.path.basename(p)} -> step={step_from_ckpt(p)}")

    import wandb
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    # When attaching to the training run (--wandb_run_id set), don't pass
    # `name` — wandb would otherwise overwrite the run's original name.
    init_kwargs = dict(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=args.wandb_run_id,
        resume="must" if args.wandb_run_id else None,
        tags=[t.strip() for t in (args.wandb_tags or "").split(",") if t.strip()] or None,
    )
    if not args.wandb_run_id:
        init_kwargs["name"] = (args.wandb_run_name or
                               os.path.basename(args.ckpt_dir.rstrip("/")) + "-inf-eval")
    wandb.init(**init_kwargs)
    # Custom x-axis so HTMLs plot at the checkpoint's training step regardless
    # of where wandb's monotonic internal step is (esp. when resuming a run).
    wandb.define_metric("eval/inf/ckpt_step")
    wandb.define_metric("eval/inf/samples", step_metric="eval/inf/ckpt_step")

    cb = InferenceEvalCallback(
        eval_hf_dataset=eval_hf_dataset,
        processor=processor,
        n_samples=args.inf_eval_samples,
        max_n_images=args.max_n_images_train,
        max_new_tokens=args.inf_eval_max_new_tokens,
        seed=args.inf_eval_seed,
    )

    if args.include_base_model:
        print(f"\n[inf-eval] === BASE MODEL {args.base_model_id} (step=0) ===")
        model = load_model(
            args.base_model_id, args.base_model_id, dtype,
            args.attn_implementation, device,
        )
        try:
            cb.on_evaluate(
                args=None,
                state=SimpleNamespace(global_step=0),
                control=None,
                model=model,
            )
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    for ckpt in checkpoints:
        step = step_from_ckpt(ckpt)
        print(f"\n[inf-eval] === {os.path.basename(ckpt)} (step={step}) ===")
        model = load_model(
            ckpt, args.base_model_id, dtype, args.attn_implementation, device,
        )
        try:
            cb.on_evaluate(
                args=None,
                state=SimpleNamespace(global_step=step),
                control=None,
                model=model,
            )
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    wandb.finish()
    print("\n[inf-eval] done")


if __name__ == "__main__":
    main()
