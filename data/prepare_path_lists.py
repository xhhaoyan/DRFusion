"""Prepare generic path-list files for DRFusion training.

The script expects each video to use either:

  video_name/visible and video_name/infrared

or a legacy naming convention:

  video_name/channel  and video_name/channel2

where ``channel`` is visible and ``channel2`` is infrared.
"""

import argparse
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    if not folder.exists():
        return []
    return sorted(
        str(path)
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_pair_dirs(video_dir):
    visible_dir = video_dir / "visible"
    infrared_dir = video_dir / "infrared"
    if visible_dir.exists() and infrared_dir.exists():
        return visible_dir, infrared_dir

    visible_dir = video_dir / "channel"
    infrared_dir = video_dir / "channel2"
    if visible_dir.exists() and infrared_dir.exists():
        return visible_dir, infrared_dir

    return None, None


def collect_videos(root):
    videos = []
    for video_dir in sorted(root.iterdir()):
        if not video_dir.is_dir():
            continue

        visible_dir, infrared_dir = find_pair_dirs(video_dir)
        if visible_dir is None:
            continue

        visible = list_images(visible_dir)
        infrared = list_images(infrared_dir)
        if not visible or not infrared:
            continue

        videos.append(
            {
                "name": video_dir.name,
                "visible": visible,
                "infrared": infrared,
                "representative": visible[0],
            }
        )

    return videos


def write_lines(lines, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(str(line) + "\n")
    print(f"Saved {len(lines)} paths to {output_file}")


def flatten(videos, key):
    paths = []
    for video in videos:
        paths.extend(video[key])
    return paths


def split_videos(videos, train_ratio, seed):
    random.Random(seed).shuffle(videos)
    split_idx = max(1, int(len(videos) * train_ratio))
    return sorted(videos[:split_idx], key=lambda item: item["name"]), sorted(
        videos[split_idx:],
        key=lambda item: item["name"],
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare DRFusion path-list files.")
    parser.add_argument("--root", required=True, help="Dataset root containing video folders.")
    parser.add_argument("--output_dir", default="data", help="Directory for generated txt files.")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output_dir)

    videos = collect_videos(root)
    if not videos:
        raise ValueError(f"No valid paired videos found under {root}")

    train_videos, val_videos = split_videos(videos, args.train_ratio, args.seed)
    print(f"Found {len(videos)} videos: train={len(train_videos)}, val={len(val_videos)}")

    train_visible = flatten(train_videos, "visible")
    val_visible = flatten(val_videos, "visible")
    train_infrared = flatten(train_videos, "infrared")
    val_infrared = flatten(val_videos, "infrared")

    write_lines(train_visible, output_dir / "visible_train_paths.txt")
    write_lines(val_visible, output_dir / "visible_val_paths.txt")
    write_lines(train_infrared, output_dir / "infrared_train_paths.txt")
    write_lines(val_infrared, output_dir / "infrared_val_paths.txt")
    write_lines(train_visible + train_infrared, output_dir / "all_train_paths.txt")
    write_lines(val_visible + val_infrared, output_dir / "all_val_paths.txt")
    write_lines([item["representative"] for item in train_videos], output_dir / "video_train_paths.txt")
    write_lines([item["representative"] for item in val_videos], output_dir / "video_val_paths.txt")


if __name__ == "__main__":
    main()
