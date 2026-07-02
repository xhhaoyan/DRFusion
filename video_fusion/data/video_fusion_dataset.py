"""
Video Fusion Dataset for IR + VI sequences
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class VideoFusionDataset(Dataset):
    """
    Dataset for video fusion tasks with infrared and visible light pairs.

    Expected directory structure:
        root_dir/
            video_1/
                infrared/
                    frame001.jpg
                    frame002.jpg
                    ...
                visible/
                    frame001.jpg
                    frame002.jpg
                    ...
            video_2/
                ...
    """

    def __init__(
        self,
        root_dir,
        n_frames=16,
        frame_skip=1,
        img_size=256,
        return_video_name=True,
        load_all_frames=False,
        filter_video_names=None
    ):
        """
        Args:
            root_dir: Root directory containing video sequences
            n_frames: Number of frames to sample (ignored if load_all_frames=True)
            frame_skip: Skip frames for temporal downsampling
            img_size: Resize images to this size
            return_video_name: Whether to return video name in output
            load_all_frames: If True, load all frames in video (ignores n_frames)
            filter_video_names: Optional set of video names to include (for train/val split)
        """
        self.root_dir = root_dir
        self.n_frames = n_frames
        self.frame_skip = frame_skip
        self.img_size = (img_size, img_size)
        self.return_video_name = return_video_name
        self.load_all_frames = load_all_frames
        self.filter_video_names = filter_video_names

        self.sequences = self._scan_sequences()

    def _scan_sequences(self):
        """Scan root directory for valid video sequences"""
        sequences = []

        if not os.path.exists(self.root_dir):
            raise ValueError(f"Root directory does not exist: {self.root_dir}")

        for video_name in sorted(os.listdir(self.root_dir)):
            video_path = os.path.join(self.root_dir, video_name)
            if not os.path.isdir(video_path):
                continue

            # 应用视频名称过滤（用于train/val split）
            if self.filter_video_names is not None and video_name not in self.filter_video_names:
                continue

            # Support both naming conventions:
            # 1. infrared/visible (standard)
            # 2. channel2/channel (legacy dataset naming)
            ir_path = os.path.join(video_path, 'infrared')
            vi_path = os.path.join(video_path, 'visible')

            # If standard naming does not exist, try the legacy channel naming.
            if not os.path.exists(ir_path) or not os.path.exists(vi_path):
                ir_path = os.path.join(video_path, 'channel2')  # channel2 = infrared
                vi_path = os.path.join(video_path, 'channel')   # channel = visible

            if not os.path.exists(ir_path) or not os.path.exists(vi_path):
                print(f"Warning: Skipping {video_name}, missing IR or VI folder")
                continue

            # Get frame lists
            ir_frames = sorted([f for f in os.listdir(ir_path) if f.endswith(('.jpg', '.png'))])
            vi_frames = sorted([f for f in os.listdir(vi_path) if f.endswith(('.jpg', '.png'))])

            if len(ir_frames) != len(vi_frames):
                print(f"Warning: Skipping {video_name}, frame count mismatch (IR={len(ir_frames)}, VI={len(vi_frames)})")
                continue

            # Only check frame count if not loading all frames
            if not self.load_all_frames and len(ir_frames) < self.n_frames * self.frame_skip:
                print(f"Warning: Skipping {video_name}, not enough frames ({len(ir_frames)} < {self.n_frames * self.frame_skip})")
                continue

            # Get original image size from first frame
            first_frame_path = os.path.join(ir_path, ir_frames[0])
            first_frame = cv2.imread(first_frame_path)
            original_size = (first_frame.shape[1], first_frame.shape[0])  # (W, H)

            sequences.append({
                'name': video_name,
                'ir_path': ir_path,
                'vi_path': vi_path,
                'num_frames': len(ir_frames),
                'ir_frame_files': ir_frames,  # IR frame filenames
                'vi_frame_files': vi_frames,  # VI frame filenames
                'original_size': original_size  # Store original size
            })

        print(f"Found {len(sequences)} valid video sequences")
        return sequences

    def __len__(self):
        return len(self.sequences)

    def _load_frame(self, frame_path, color_mode='gray'):
        """Load and preprocess a single frame"""
        img = cv2.imread(frame_path)
        if img is None:
            raise ValueError(f"Failed to load image: {frame_path}")

        if color_mode == 'gray':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)  # Use CUBIC for quality
            img = img.astype(np.float32) / 255.0  # [0, 1]
            img = img * 2 - 1  # [-1, 1]
            return img[..., np.newaxis]  # (H, W, 1)
        elif color_mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_CUBIC)  # Use CUBIC for quality
            img = img.astype(np.float32) / 255.0  # [0, 1]
            img = img * 2 - 1  # [-1, 1]
            return img  # (H, W, 3)
        else:
            raise ValueError(f"Unknown color mode: {color_mode}")

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - infrared: (T, 1, H, W) tensor
                - visible: (T, 3, H, W) tensor
                - video_name: str (if return_video_name=True)
                - original_size: (W, H) tuple
                - frame_files: dict with 'ir' and 'vi' lists of filenames
        """
        seq = self.sequences[idx]

        # Determine how many frames to load
        if self.load_all_frames:
            # Load all frames in video (no limit, for sliding window processing)
            num_frames_to_load = seq['num_frames']
            frame_indices = list(range(0, num_frames_to_load, self.frame_skip))
        else:
            # Sample frame indices
            max_start = seq['num_frames'] - self.n_frames * self.frame_skip
            if max_start <= 0:
                start_idx = 0
            else:
                start_idx = np.random.randint(0, max_start)

            frame_indices = [
                start_idx + i * self.frame_skip
                for i in range(self.n_frames)
            ]

        # Load frames
        ir_frames = []
        vi_frames = []

        for idx in frame_indices:
            # Get separate filenames for IR and VI
            ir_frame_file = seq['ir_frame_files'][idx]
            vi_frame_file = seq['vi_frame_files'][idx]

            # Load infrared frame (grayscale)
            ir_frame_path = os.path.join(seq['ir_path'], ir_frame_file)
            ir_frame = self._load_frame(ir_frame_path, color_mode='gray')
            ir_frames.append(ir_frame)

            # Load visible frame (RGB)
            vi_frame_path = os.path.join(seq['vi_path'], vi_frame_file)
            vi_frame = self._load_frame(vi_frame_path, color_mode='rgb')
            vi_frames.append(vi_frame)

        # Stack and convert to torch tensors
        ir_tensor = torch.from_numpy(np.stack(ir_frames)).float()  # (T, H, W, 1)
        vi_tensor = torch.from_numpy(np.stack(vi_frames)).float()  # (T, H, W, 3)

        # Permute to (T, C, H, W)
        ir_tensor = ir_tensor.permute(0, 3, 1, 2)
        vi_tensor = vi_tensor.permute(0, 3, 1, 2)

        # Convert original_size to tensor for DataLoader compatibility
        original_size_tensor = torch.tensor(seq['original_size'], dtype=torch.long)  # (W, H)

        # Store frame filenames for debugging (use VI filenames as reference)
        frame_filenames = {
            'ir': [seq['ir_frame_files'][idx] for idx in frame_indices],
            'vi': [seq['vi_frame_files'][idx] for idx in frame_indices]
        }

        result = {
            'infrared': ir_tensor,
            'visible': vi_tensor,
            'original_size': original_size_tensor,
            'frame_files': frame_filenames  # Store both IR and VI filenames
        }

        if self.return_video_name:
            result['video_name'] = seq['name']

        return result
