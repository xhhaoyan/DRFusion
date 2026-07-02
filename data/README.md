# Data Lists

This directory contains dataset helpers only. Local path-list files are not committed because they contain machine-specific paths.

Generate paired video lists with:

```bash
python data/prepare_path_lists.py \
  --root datasets/train_videos \
  --output_dir data
```

The command creates the files referenced by the training configs, including:

- `visible_train_paths.txt`
- `visible_val_paths.txt`
- `video_train_paths.txt`
- `video_val_paths.txt`
