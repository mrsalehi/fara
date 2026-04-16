"""
Adaptive patching for web agent trajectories.

Uses cross-correlation to detect vertical scrolling between frames,
then hierarchical diff to efficiently find changed regions:

  1. Estimate scroll offset via cross-correlation on a vertical pixel strip
  2. Shift previous frame by that offset to align content
  3. Hierarchical diff: compare at 224x224 first; only split blocks that
     changed into 112→56→28 sub-patches. Unchanged blocks stay as one
     large patch (= 1 token).
  4. Frame 0 gets intra-frame quadtree based on variance.

Leading blank frames and trailing duplicate frames are automatically dropped.

Input:  parquet file with columns [sample_id, instruction, trajectory, images]
        where 'images' is an array of dicts with 'bytes' key (PNG-encoded).

Usage:
  # Stats only
  python trajectory_patch.py /path/to/data.parquet --num-trajs 50

  # Stats + visualizations
  python trajectory_patch.py /path/to/data.parquet --num-trajs 30 --viz -o viz_output/

  # All parquet files in a directory
  python trajectory_patch.py --data-dir /path/to/data/ --num-trajs 0 --stats-csv stats.csv
"""

import argparse
import io
import multiprocessing as mp
import os

import numpy as np
from scipy.signal import correlate
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Quadtree config
# ---------------------------------------------------------------------------
PATCH_SIZES = [224, 112, 56, 28]

BORDER_COLORS = {
    224: (255, 0, 0, 255),
    112: (0, 255, 0, 255),
    56:  (0, 0, 255, 255),
    28:  (255, 165, 0, 255),
}
FILL_COLORS = {
    224: (255, 0, 0, 30),
    112: (0, 255, 0, 30),
    56:  (0, 0, 255, 30),
    28:  (255, 165, 0, 30),
}
# Visualization: dropped patches get a cross-hatch pattern via border + fill
DROPPED_FILL = (0, 0, 0, 150)
DROPPED_BORDER = (255, 255, 255, 200)


# ---------------------------------------------------------------------------
# Scroll detection via cross-correlation
# ---------------------------------------------------------------------------

def estimate_scroll_offset(prev_arr, curr_arr, max_scroll=None):
    """
    Estimate vertical scroll offset between two frames using
    cross-correlation on a vertical pixel strip.

    Returns dy such that prev_arr shifted down by dy aligns with curr_arr.
    Positive dy = page scrolled down (content moved up).
    """
    # Different frame sizes means page navigation, not scroll
    if prev_arr.shape[:2] != curr_arr.shape[:2]:
        return 0

    h = prev_arr.shape[0]
    if max_scroll is None:
        max_scroll = h  # allow full-frame scroll

    # Use the middle column, convert to grayscale for speed
    mid_x = prev_arr.shape[1] // 2
    strip_prev = prev_arr[:, mid_x, :].mean(axis=-1).astype(np.float32)
    strip_curr = curr_arr[:, mid_x, :].mean(axis=-1).astype(np.float32)

    # Normalize to zero-mean to avoid DC bias
    strip_prev -= strip_prev.mean()
    strip_curr -= strip_curr.mean()

    # Cross-correlate
    corr = correlate(strip_prev, strip_curr, mode='full')
    # corr[h-1] corresponds to zero offset
    center = h - 1

    # Restrict search to [-max_scroll, +max_scroll]
    lo = max(0, center - max_scroll)
    hi = min(len(corr), center + max_scroll + 1)
    best_idx = lo + np.argmax(corr[lo:hi])
    dy = best_idx - center

    # Confidence: is the peak significantly above the noise?
    peak_val = corr[best_idx]
    corr_std = np.std(corr[lo:hi])
    confident = peak_val > 3 * corr_std if corr_std > 0 else True

    return dy if confident else 0


def shift_frame(arr, dy):
    """
    Shift a frame vertically by dy pixels.
    Positive dy = content moved up (scroll down), so we shift the array up.
    Uncovered regions are filled with zeros.
    """
    if dy == 0:
        return arr.copy()
    shifted = np.zeros_like(arr)
    h = arr.shape[0]
    if dy > 0:
        # Scrolled down: content moved up, so prev[dy:] aligns with curr[:-dy]
        if dy < h:
            shifted[:h-dy] = arr[dy:]
    else:
        # Scrolled up: content moved down
        ady = -dy
        if ady < h:
            shifted[ady:] = arr[:h-ady]
    return shifted


# ---------------------------------------------------------------------------
# Intra-frame quadtree (for frame 0 only)
# ---------------------------------------------------------------------------

def compute_variance(arr, x, y, size):
    patch = arr[y:y+size, x:x+size]
    if patch.size == 0:
        return 0.0
    return float(np.var(patch.astype(np.float32)))


def quadtree_decompose(arr, x, y, size, var_thresh):
    h, w = arr.shape[:2]
    if min(size, w - x) <= 0 or min(size, h - y) <= 0:
        return []

    var = compute_variance(arr, x, y, size)
    idx = PATCH_SIZES.index(size)
    can_split = idx < len(PATCH_SIZES) - 1

    if var > var_thresh and can_split:
        child = PATCH_SIZES[idx + 1]
        patches = []
        for dy in range(0, size, child):
            for dx in range(0, size, child):
                cx, cy = x + dx, y + dy
                if cx < w and cy < h:
                    patches.extend(quadtree_decompose(arr, cx, cy, child, var_thresh))
        return patches
    return [(x, y, size)]


def decompose_frame(arr, var_thresh):
    h, w = arr.shape[:2]
    largest = PATCH_SIZES[0]
    patches = []
    for y in range(0, h, largest):
        for x in range(0, w, largest):
            patches.extend(quadtree_decompose(arr, x, y, largest, var_thresh))
    return patches


# ---------------------------------------------------------------------------
# Hierarchical diff (for frames t>0)
# ---------------------------------------------------------------------------

def patch_mse(arr1, arr2, x, y, size):
    """MSE between the same region in two frames."""
    p1 = arr1[y:y+size, x:x+size].astype(np.float32)
    p2 = arr2[y:y+size, x:x+size].astype(np.float32)
    if p1.shape != p2.shape:
        return float('inf')
    return float(np.mean((p1 - p2) ** 2))


def hierarchical_diff(curr_arr, prev_aligned, x, y, size, mse_thresh,
                      var_thresh):
    """
    Compare a block between curr and scroll-aligned prev.
    If unchanged → drop (one large patch).
    If changed but uniform in curr frame (low variance) → keep as one patch.
    If changed and complex → recurse on children.
    If changed and at smallest size → keep.

    Returns (kept, dropped) lists of (x, y, size) tuples.
    """
    h, w = curr_arr.shape[:2]
    if min(size, w - x) <= 0 or min(size, h - y) <= 0:
        return [], []

    mse = patch_mse(curr_arr, prev_aligned, x, y, size)

    if mse < mse_thresh:
        # Unchanged — drop as one patch
        return [], [(x, y, size)]

    # Changed — check if the region is uniform in the current frame.
    # If so, keep as one large patch (e.g. white space that appeared).
    var = compute_variance(curr_arr, x, y, size)
    if var <= var_thresh:
        return [(x, y, size)], []

    # Changed and complex — try to split
    idx = PATCH_SIZES.index(size)
    can_split = idx < len(PATCH_SIZES) - 1

    if can_split:
        child = PATCH_SIZES[idx + 1]
        kept, dropped = [], []
        for cdy in range(0, size, child):
            for cdx in range(0, size, child):
                cx, cy = x + cdx, y + cdy
                if cx < w and cy < h:
                    k, d = hierarchical_diff(
                        curr_arr, prev_aligned, cx, cy, child, mse_thresh,
                        var_thresh)
                    kept.extend(k)
                    dropped.extend(d)
        return kept, dropped
    else:
        # Smallest level, changed — keep it
        return [(x, y, size)], []


def diff_frame(curr_arr, prev_aligned, mse_thresh, var_thresh):
    """
    Hierarchical diff of entire frame against scroll-aligned previous frame.
    Returns (kept_patches, dropped_patches).
    """
    h, w = curr_arr.shape[:2]
    largest = PATCH_SIZES[0]
    kept_all, dropped_all = [], []
    for y in range(0, h, largest):
        for x in range(0, w, largest):
            k, d = hierarchical_diff(
                curr_arr, prev_aligned, x, y, largest, mse_thresh,
                var_thresh)
            kept_all.extend(k)
            dropped_all.extend(d)
    return kept_all, dropped_all


# ---------------------------------------------------------------------------
# Process a full trajectory
# ---------------------------------------------------------------------------

def is_blank_frame(arr, max_var=1.0):
    """Check if a frame is essentially blank (all white, all black, etc.)."""
    return float(np.var(arr.astype(np.float32))) <= max_var


def frames_identical(arr1, arr2):
    """Check if two frames are pixel-identical."""
    if arr1.shape != arr2.shape:
        return False
    return np.array_equal(arr1, arr2)


def process_trajectory(images, var_thresh=100.0, mse_thresh=10.0):
    """
    Process all frames.

    Frame 0: intra-frame quadtree decomposition (based on variance).
    Frame t>0: estimate scroll offset via cross-correlation, shift previous
    frame to align, then hierarchical diff (compare-first, split-only-if-changed).

    Skips leading blank frames and drops trailing duplicate frames.

    Returns (results, n_skipped_blank, trailing_dup_dropped).
    Each result dict has: frame_idx, kept_patches, dropped_patches,
    img_size, img, scroll_dy.
    """
    # Load all frames upfront so we can inspect first/last
    frames = []
    for item in images:
        img = load_frame(item)
        frames.append((img, np.array(img)))

    # Skip leading blank frames
    n_skipped_blank = 0
    while n_skipped_blank < len(frames) and is_blank_frame(frames[n_skipped_blank][1]):
        n_skipped_blank += 1

    # Drop trailing duplicate frame
    trailing_dup_dropped = False
    if len(frames) - n_skipped_blank >= 2:
        last_arr = frames[-1][1]
        penult_arr = frames[-2][1]
        if frames_identical(last_arr, penult_arr):
            trailing_dup_dropped = True

    end = len(frames) - (1 if trailing_dup_dropped else 0)
    active_frames = frames[n_skipped_blank:end]

    results = []
    prev_arr = None

    for rel_i, (img, arr) in enumerate(active_frames):
        orig_idx = n_skipped_blank + rel_i

        if prev_arr is None:
            # Frame 0: intra-frame quadtree
            all_patches = decompose_frame(arr, var_thresh)
            results.append({
                'frame_idx': orig_idx,
                'kept_patches': all_patches,
                'dropped_patches': [],
                'img_size': img.size,
                'img': img,
                'scroll_dy': 0,
            })
        else:
            # Estimate scroll and align
            dy = estimate_scroll_offset(prev_arr, arr)
            prev_shifted = shift_frame(prev_arr, dy)

            # If frame sizes differ, create a matching-size aligned frame
            # (zeros where no previous data exists → those regions count as changed)
            h_curr, w_curr = arr.shape[:2]
            h_prev, w_prev = prev_shifted.shape[:2]
            if (h_curr, w_curr) != (h_prev, w_prev):
                prev_aligned = np.zeros_like(arr)
                h_min, w_min = min(h_curr, h_prev), min(w_curr, w_prev)
                prev_aligned[:h_min, :w_min] = prev_shifted[:h_min, :w_min]
            else:
                prev_aligned = prev_shifted

            # Hierarchical diff against scroll-aligned previous frame
            kept, dropped = diff_frame(arr, prev_aligned, mse_thresh, var_thresh)

            results.append({
                'frame_idx': orig_idx,
                'kept_patches': kept,
                'dropped_patches': dropped,
                'img_size': img.size,
                'img': img,
                'scroll_dy': dy,
            })

        prev_arr = arr

    return results, n_skipped_blank, trailing_dup_dropped


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_frame(item):
    if isinstance(item, np.ndarray):
        arr = item
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert('RGB')
    if isinstance(item, Image.Image):
        return item.convert('RGB')
    if isinstance(item, dict):
        raw = item['bytes']
    else:
        raw = item
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    elif isinstance(raw, bytearray):
        raw = bytes(raw)
    return Image.open(io.BytesIO(raw)).convert('RGB')


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def trajectory_stats(results, n_skipped_blank=0, trailing_dup_dropped=False,
                     n_original_frames=None):
    if not results:
        return {}

    w, h = results[0]['img_size']
    # Baseline: every patch is 28x28, each patch = 1 token
    uniform_per_frame = (w // 28) * (h // 28)

    n_orig = n_original_frames if n_original_frames is not None else len(results)
    baseline_tokens = uniform_per_frame * n_orig

    # Our method: each patch = 1 token regardless of size, dropped patches = 0
    total_kept = sum(len(r['kept_patches']) for r in results)
    total_dropped = sum(len(r['dropped_patches']) for r in results)
    scrolls = [r['scroll_dy'] for r in results if r['scroll_dy'] != 0]

    return {
        'n_original_frames': n_orig,
        'n_active_frames': len(results),
        'n_skipped_blank': n_skipped_blank,
        'trailing_dup_dropped': trailing_dup_dropped,
        'img_size': f'{w}x{h}',
        'baseline_tokens': baseline_tokens,
        'final_tokens': total_kept,
        'reduction_pct': (1 - total_kept / baseline_tokens) * 100 if baseline_tokens else 0,
        'kept_patches': total_kept,
        'dropped_patches': total_dropped,
        'n_scroll_frames': len(scrolls),
        'scroll_offsets': scrolls,
    }


def print_trajectory_stats(stats, traj_idx=None):
    prefix = f"Traj {traj_idx}: " if traj_idx is not None else ""
    skipped_parts = []
    if stats['n_skipped_blank'] > 0:
        skipped_parts.append(f"{stats['n_skipped_blank']} blank")
    if stats['trailing_dup_dropped']:
        skipped_parts.append("1 trailing dup")
    skipped_str = f" (skipped: {', '.join(skipped_parts)})" if skipped_parts else ""
    scroll_str = ""
    if stats['n_scroll_frames'] > 0:
        scroll_str = f", {stats['n_scroll_frames']} scrolls detected"
    print(f"{prefix}{stats['n_original_frames']} frames → "
          f"{stats['n_active_frames']} active, {stats['img_size']}"
          f"{skipped_str}{scroll_str}")
    print(f"  Baseline (uniform 28x28): {stats['baseline_tokens']} tokens")
    print(f"  Ours (hierarchical diff): {stats['final_tokens']} tokens "
          f"({stats['reduction_pct']:.1f}% reduction)")
    print(f"  Patches: {stats['kept_patches']} kept, "
          f"{stats['dropped_patches']} dropped")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_frame(img, kept, dropped, frame_idx, scroll_dy, output_path):
    """
    Show original image with:
    - Kept patches: color-coded borders by size (red/green/blue/orange)
    - Dropped patches: dark overlay (unchanged from previous frame)
    """
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw dropped patches: dark fill + diagonal line so they're visible on any bg
    for (x, y, size) in dropped:
        draw.rectangle([x, y, x+size-1, y+size-1],
                       fill=DROPPED_FILL, outline=DROPPED_BORDER, width=1)
        draw.line([(x, y), (x+size-1, y+size-1)],
                  fill=(255, 255, 255, 140), width=1)

    # Draw kept patches with color borders
    for (x, y, size) in kept:
        draw.rectangle([x, y, x+size-1, y+size-1],
                       fill=FILL_COLORS[size], outline=BORDER_COLORS[size], width=1)

    result = Image.alpha_composite(img.convert("RGBA"), overlay)

    # Header label
    label_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label_overlay)
    scroll_str = f"  scroll: {scroll_dy:+d}px" if scroll_dy != 0 else ""
    text = (f"Frame {frame_idx}  |  "
            f"kept: {len(kept)}  |  "
            f"dropped: {len(dropped)}{scroll_str}")
    label_draw.rectangle([0, 0, len(text) * 7 + 10, 18], fill=(0, 0, 0, 200))
    label_draw.text((5, 2), text, fill=(255, 255, 255, 255))
    result = Image.alpha_composite(result, label_overlay)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    result.save(output_path)


def visualize_trajectory(df_row, row_idx, output_dir, var_thresh, mse_thresh):
    images = df_row['images']
    items = images.tolist() if isinstance(images, np.ndarray) else list(images)
    n_original = len(items)
    results, n_skipped, trailing_dup = process_trajectory(items, var_thresh, mse_thresh)

    traj_dir = os.path.join(output_dir, f"traj_{row_idx:04d}")
    for r in results:
        out_path = os.path.join(traj_dir, f"frame_{r['frame_idx']:03d}.png")
        visualize_frame(r['img'], r['kept_patches'], r['dropped_patches'],
                        r['frame_idx'], r['scroll_dy'], out_path)

    stats = trajectory_stats(results, n_skipped, trailing_dup, n_original)
    print_trajectory_stats(stats, traj_idx=row_idx)
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage adaptive patching for web agent trajectories")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("parquet", nargs="?", default=None,
                             help="Path to a single parquet file")
    input_group.add_argument("--data-dir",
                             help="Path to directory; processes all .parquet files in it")
    parser.add_argument("--num-trajs", type=int, default=10,
                        help="Number of trajectories to process per file for stats")
    parser.add_argument("--viz", action="store_true",
                        help="Save visualizations for all processed trajectories")
    parser.add_argument("-o", "--output", default="traj_patch_viz",
                        help="Output directory for visualizations")
    parser.add_argument("--var-thresh", type=float, default=100.0,
                        help="Variance threshold for quadtree splitting")
    parser.add_argument("--mse-thresh", type=float, default=10.0,
                        help="MSE threshold for unchanged detection")
    parser.add_argument("--stats-csv", help="Write per-trajectory stats to CSV")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1, 0 = all CPUs)")
    args = parser.parse_args()

    # Collect parquet files
    if args.data_dir:
        import glob
        parquet_files = sorted(glob.glob(os.path.join(args.data_dir, "*.parquet")))
        if not parquet_files:
            print(f"No parquet files found in {args.data_dir}")
            return
        print(f"Found {len(parquet_files)} parquet files in {args.data_dir}\n")
    else:
        parquet_files = [args.parquet]

    all_stats = []

    file_pbar = tqdm(parquet_files, desc="Files", unit="file")
    for pq_path in file_pbar:
        fname = os.path.basename(pq_path)
        file_pbar.set_postfix_str(fname)
        df = pd.read_parquet(pq_path)

        n = len(df) if args.num_trajs == 0 else min(args.num_trajs, len(df))
        traj_pbar = tqdm(range(n), desc=f"  {fname}", unit="traj", leave=False)
        for i in traj_pbar:
            row = df.iloc[i]
            images = row['images']
            items = images.tolist() if isinstance(images, np.ndarray) else list(images)
            n_original = len(items)
            results, n_skipped, trailing_dup = process_trajectory(
                items, args.var_thresh, args.mse_thresh)
            s = trajectory_stats(results, n_skipped, trailing_dup, n_original)
            if not s:
                continue
            s['traj_idx'] = i
            s['file'] = fname
            all_stats.append(s)

            if args.viz:
                # file stem without extension as subdirectory
                file_stem = os.path.splitext(fname)[0]
                traj_dir = os.path.join(args.output, file_stem, f"traj_{i:04d}")
                for r in results:
                    out_path = os.path.join(traj_dir, f"frame_{r['frame_idx']:03d}.png")
                    visualize_frame(r['img'], r['kept_patches'], r['dropped_patches'],
                                    r['frame_idx'], r['scroll_dy'], out_path)
        traj_pbar.close()

    # Aggregates — filter out empty stats (trajectories with no active frames)
    all_stats = [s for s in all_stats if s]
    if all_stats:
        total_baseline = sum(s['baseline_tokens'] for s in all_stats)
        total_final = sum(s['final_tokens'] for s in all_stats)
        avg_red = np.mean([s['reduction_pct'] for s in all_stats])
        print("=" * 60)
        print(f"Aggregate over {len(all_stats)} trajectories "
              f"from {len(parquet_files)} file(s):")
        print(f"  {total_baseline} → {total_final} tokens "
              f"({(1 - total_final/total_baseline)*100:.1f}% total, "
              f"{avg_red:.1f}% avg per traj)")

    if args.stats_csv and all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(args.stats_csv, index=False)
        print(f"\nStats written to {args.stats_csv}")


if __name__ == "__main__":
    main()
