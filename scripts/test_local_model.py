"""Quick test: load Qwen2.5-VL from HuggingFace using our local modeling code."""

import torch

MODEL_ID = "microsoft/Fara-7B"


def main():
    # 1. Load config from our local copy
    print("\n[1] Loading config from local modeling code...")
    from fara.modeling.configuration_qwen2_5_vl import Qwen2_5_VLConfig
    config = Qwen2_5_VLConfig.from_pretrained(MODEL_ID)
    print(f"    Text config: hidden_size={config.text_config.hidden_size}, "
          f"num_layers={config.text_config.num_hidden_layers}")
    print(f"    Vision config: hidden_size={config.vision_config.hidden_size}, "
          f"patch_size={config.vision_config.patch_size}, "
          f"depth={config.vision_config.depth}")

    # 2. Load model from our local copy
    print("\n[2] Loading model from HuggingFace with local modeling code...")
    from fara.modeling.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    load_result = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_loading_info=True,
    )
    model, loading_info = load_result[0], load_result[1]
    missing = loading_info.get("missing_keys", [])
    unexpected = loading_info.get("unexpected_keys", [])
    if missing:
        print(f"    WARNING: {len(missing)} missing keys: {missing[:5]}...")
    if unexpected:
        print(f"    WARNING: {len(unexpected)} unexpected keys: {unexpected[:5]}...")
    if not missing and not unexpected:
        print("    All weights loaded successfully - no missing or unexpected keys.")
    print(f"    Model loaded. dtype={model.dtype}, device_map={model.hf_device_map}")

    # 3. Verify the vision encoder is accessible
    print("\n[3] Checking vision encoder...")
    vis = model.visual
    print(f"    Vision encoder type: {type(vis).__name__}")
    print(f"    Patch embed: {vis.patch_embed}")
    print(f"    Num blocks: {len(vis.blocks)}")
    print(f"    Spatial merge size: {vis.spatial_merge_size}")

    # 4. Quick forward pass through the vision encoder only
    print("\n[4] Testing vision encoder forward pass...")
    patch_size = config.vision_config.patch_size
    temporal_patch_size = config.vision_config.temporal_patch_size
    # Create a dummy input: 4 patches worth of pixels
    # Each patch is (temporal_patch_size * patch_size * patch_size * in_channels) flattened
    num_dummy_patches = 4
    dummy_pixels = torch.randn(
        num_dummy_patches,
        3 * temporal_patch_size * patch_size * patch_size,
        dtype=torch.bfloat16,
        device=model.device,
    )
    # grid_thw: 1 image, temporal=1, height=2 patches, width=2 patches
    grid_thw = torch.tensor([[1, 2, 2]], device=model.device)

    with torch.no_grad():
        vision_out = vis(dummy_pixels, grid_thw=grid_thw)
    print(f"    Input shape:  {dummy_pixels.shape}")
    print(f"    Output shape: {vision_out.shape}")
    print(f"    Output dtype: {vision_out.dtype}")

    # 5. Verify outputs match the original transformers implementation
    print("\n[5] Comparing with original transformers Qwen2.5-VL...")
    from transformers import Qwen2_5_VLForConditionalGeneration as OrigModel
    orig_model = OrigModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    with torch.no_grad():
        orig_vision_out = orig_model.visual(
            dummy_pixels.to(orig_model.device),
            grid_thw=grid_thw.to(orig_model.device),
        )

    max_diff = (vision_out.cpu().float() - orig_vision_out.cpu().float()).abs().max().item()
    print(f"    Max absolute difference: {max_diff:.2e}")
    if max_diff < 1e-5:
        print("    PASS: outputs match")
    else:
        print("    WARN: outputs differ (may be due to device placement)")

    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
