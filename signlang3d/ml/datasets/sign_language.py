"""
Sign Language Dataset for PyTorch

This module provides dataset classes for loading and preprocessing
3D skeletal gesture data for sign language translation.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class SignLanguageDataset(Dataset):
    """
    PyTorch Dataset for sign language gesture sequences.
    
    Loads skeletal motion data and corresponding text transcripts
    for training translation models.
    """
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        samples: Optional[List[Dict]] = None,
        tokenizer: Optional[Any] = None,
        max_frames: int = 150,
        max_text_length: int = 100,
        num_joints: int = 21,
        augment: bool = False,
        normalize: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing JSON sample files
            samples: List of sample dictionaries (alternative to data_dir)
            tokenizer: Text tokenizer for transcripts
            max_frames: Maximum number of frames per sample
            max_text_length: Maximum text token length
            num_joints: Number of skeleton joints
            augment: Whether to apply data augmentation
            normalize: Whether to normalize skeleton coordinates
        """
        self.max_frames = max_frames
        self.max_text_length = max_text_length
        self.num_joints = num_joints
        self.augment = augment
        self.normalize = normalize
        self.tokenizer = tokenizer
        
        # Load samples
        if samples is not None:
            self.samples = samples
        elif data_dir is not None:
            self.samples = self._load_from_directory(Path(data_dir))
        else:
            self.samples = []
        
        # Compute normalization statistics
        if normalize and self.samples:
            self.mean, self.std = self._compute_normalization_stats()
        else:
            self.mean = np.zeros(3)
            self.std = np.ones(3)
    
    def _load_from_directory(self, data_dir: Path) -> List[Dict]:
        """Load samples from JSON files in directory."""
        samples = []
        
        for json_file in data_dir.glob('*.json'):
            with open(json_file, 'r') as f:
                sample = json.load(f)
                samples.append(sample)
        
        return samples
    
    def _compute_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and std for normalization."""
        all_coords = []
        
        for sample in self.samples[:100]:  # Use subset for efficiency
            frames = sample.get('frames', [])
            for frame in frames:
                for joint in frame:
                    all_coords.append([joint['x'], joint['y'], joint['z']])
        
        if not all_coords:
            return np.zeros(3), np.ones(3)
        
        all_coords = np.array(all_coords)
        return all_coords.mean(axis=0), all_coords.std(axis=0) + 1e-6
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
            - skeleton: (T, V, C) tensor of joint positions
            - skeleton_mask: (T,) boolean mask (True = valid)
            - text_ids: (S,) tensor of token IDs
            - text_mask: (S,) boolean mask
            - language: language code
            - gloss: sign gloss notation
        """
        sample = self.samples[idx]
        
        # Parse skeleton data
        skeleton = self._parse_skeleton(sample['frames'])
        
        # Apply augmentation if enabled
        if self.augment:
            skeleton = self._augment_skeleton(skeleton)
        
        # Normalize
        if self.normalize:
            skeleton = (skeleton - self.mean) / self.std
        
        # Truncate or pad skeleton
        skeleton, skeleton_mask = self._pad_skeleton(skeleton)
        
        # Tokenize text
        transcript = sample.get('transcript', '')
        if self.tokenizer is not None:
            text_ids = self.tokenizer.encode(
                transcript,
                add_special_tokens=True,
                max_length=self.max_text_length,
                truncation=True
            )
            text_ids = torch.tensor(text_ids, dtype=torch.long)
        else:
            # Placeholder: character-level encoding
            text_ids = torch.tensor(
                [ord(c) for c in transcript[:self.max_text_length]],
                dtype=torch.long
            )
        
        return {
            'skeleton': torch.tensor(skeleton, dtype=torch.float32),
            'skeleton_mask': torch.tensor(skeleton_mask, dtype=torch.bool),
            'text_ids': text_ids,
            'language': sample.get('language', 'unknown'),
            'gloss': sample.get('gloss', ''),
            'transcript': transcript,
        }
    
    def _parse_skeleton(self, frames: List[List[Dict]]) -> np.ndarray:
        """
        Parse frame data into numpy array.
        
        Args:
            frames: List of frames, each containing list of joint dicts
        
        Returns:
            Array of shape (T, V, 3)
        """
        num_frames = len(frames)
        skeleton = np.zeros((num_frames, self.num_joints, 3))
        
        # Create joint name to index mapping
        joint_names = [
            'wrist',
            'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
            'index_mcp', 'index_pip', 'index_dip', 'index_tip',
            'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
            'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
            'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'
        ]
        joint_to_idx = {name: i for i, name in enumerate(joint_names)}
        
        for t, frame in enumerate(frames):
            for joint in frame:
                joint_name = joint.get('joint', '')
                
                # Try exact match first
                if joint_name in joint_to_idx:
                    idx = joint_to_idx[joint_name]
                else:
                    # Try to find by index if provided
                    idx = joint.get('index', None)
                    if idx is None or idx >= self.num_joints:
                        continue
                
                skeleton[t, idx, 0] = joint.get('x', 0)
                skeleton[t, idx, 1] = joint.get('y', 0)
                skeleton[t, idx, 2] = joint.get('z', 0)
        
        return skeleton
    
    def _pad_skeleton(
        self,
        skeleton: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad or truncate skeleton sequence.
        
        Returns:
            Tuple of (padded skeleton, mask)
        """
        T = skeleton.shape[0]
        
        if T > self.max_frames:
            # Truncate
            skeleton = skeleton[:self.max_frames]
            mask = np.ones(self.max_frames, dtype=bool)
        else:
            # Pad
            padding = np.zeros((self.max_frames - T, self.num_joints, 3))
            skeleton = np.concatenate([skeleton, padding], axis=0)
            mask = np.zeros(self.max_frames, dtype=bool)
            mask[:T] = True
        
        return skeleton, mask
    
    def _augment_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """Apply data augmentation to skeleton."""
        # Random rotation around vertical axis
        if np.random.random() < 0.5:
            angle = np.random.uniform(-np.pi / 6, np.pi / 6)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
            skeleton = skeleton @ rotation
        
        # Random scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            skeleton = skeleton * scale
        
        # Random translation
        if np.random.random() < 0.5:
            translation = np.random.uniform(-0.1, 0.1, size=3)
            skeleton = skeleton + translation
        
        # Random temporal jittering
        if np.random.random() < 0.3:
            T = skeleton.shape[0]
            indices = np.clip(
                np.arange(T) + np.random.randint(-1, 2, T),
                0, T - 1
            )
            skeleton = skeleton[indices]
        
        return skeleton


class SignLanguageCollator:
    """
    Custom collator for batching sign language samples.
    
    Handles variable-length sequences with proper padding.
    """
    
    def __init__(
        self,
        pad_token_id: int = 0,
        max_frames: int = 150,
        max_text_length: int = 100
    ):
        self.pad_token_id = pad_token_id
        self.max_frames = max_frames
        self.max_text_length = max_text_length
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.
        
        Args:
            batch: List of sample dictionaries
        
        Returns:
            Batched dictionary with padded tensors
        """
        # Stack skeletons (already padded to max_frames)
        skeletons = torch.stack([s['skeleton'] for s in batch])
        skeleton_masks = torch.stack([s['skeleton_mask'] for s in batch])
        
        # Pad text sequences
        text_ids = [s['text_ids'] for s in batch]
        text_lengths = [len(t) for t in text_ids]
        max_len = max(text_lengths)
        
        text_padded = torch.full(
            (len(batch), max_len),
            self.pad_token_id,
            dtype=torch.long
        )
        text_masks = torch.zeros(len(batch), max_len, dtype=torch.bool)
        
        for i, (ids, length) in enumerate(zip(text_ids, text_lengths)):
            text_padded[i, :length] = ids
            text_masks[i, :length] = True
        
        return {
            'skeleton': skeletons,
            'skeleton_mask': skeleton_masks,
            'text_ids': text_padded,
            'text_mask': text_masks,
            'languages': [s['language'] for s in batch],
            'glosses': [s['gloss'] for s in batch],
            'transcripts': [s['transcript'] for s in batch],
        }


def create_dataloaders(
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: Optional[List[Dict]] = None,
    tokenizer: Optional[Any] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_samples: Training samples
        val_samples: Validation samples
        test_samples: Test samples (optional)
        tokenizer: Text tokenizer
        batch_size: Batch size
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional dataset arguments
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    collator = SignLanguageCollator(
        max_frames=dataset_kwargs.get('max_frames', 150),
        max_text_length=dataset_kwargs.get('max_text_length', 100)
    )
    
    train_dataset = SignLanguageDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        augment=True,
        **dataset_kwargs
    )
    
    val_dataset = SignLanguageDataset(
        samples=val_samples,
        tokenizer=tokenizer,
        augment=False,
        **dataset_kwargs
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    test_loader = None
    if test_samples:
        test_dataset = SignLanguageDataset(
            samples=test_samples,
            tokenizer=tokenizer,
            augment=False,
            **dataset_kwargs
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test with synthetic data
    synthetic_samples = [
        {
            'frames': [
                [{'joint': 'wrist', 'x': 0.1, 'y': 0.2, 'z': 0.3, 'index': 0}]
                for _ in range(60)
            ],
            'language': 'ASL',
            'gloss': 'HELLO',
            'transcript': 'Hello'
        }
        for _ in range(100)
    ]
    
    dataset = SignLanguageDataset(
        samples=synthetic_samples,
        max_frames=100,
        num_joints=21,
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Skeleton shape: {sample['skeleton'].shape}")
    print(f"Skeleton mask shape: {sample['skeleton_mask'].shape}")
    print(f"Text IDs shape: {sample['text_ids'].shape}")
