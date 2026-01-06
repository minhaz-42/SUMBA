"""
Inference Pipeline for Sign Language Translation

This module provides utilities for running inference on
trained sign language translation models.

Features:
- Single sample inference
- Batch inference
- Confidence scoring
- Beam search with multiple hypotheses
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.stgcn import STGCN
from models.motion_transformer import MotionTransformer, HybridEncoder
from models.decoder import SignLanguageTranslationModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignLanguageTranslator:
    """
    High-level interface for sign language translation.
    
    Loads a trained model and provides convenient methods
    for translating skeleton sequences to text.
    """
    
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        tokenizer: Optional[Any] = None,
        device: Optional[torch.device] = None,
        config_override: Optional[Dict] = None
    ):
        """
        Initialize translator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            tokenizer: Text tokenizer (optional)
            device: Device to run inference on
            config_override: Override checkpoint config values
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer = tokenizer
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Load checkpoint
        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device
        )
        
        # Get config
        self.config = self.checkpoint.get('config', {})
        if config_override:
            self.config.update(config_override)
        
        # Create and load model
        self.model = self._create_model()
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.eval()
        
        # Preprocessing stats
        self.mean = np.zeros(3)
        self.std = np.ones(3)
        if 'normalization' in self.checkpoint:
            self.mean = np.array(self.checkpoint['normalization']['mean'])
            self.std = np.array(self.checkpoint['normalization']['std'])
        
        logger.info(f"Loaded model from {checkpoint_path}")
        logger.info(f"Running on {self.device}")
    
    def _create_model(self) -> SignLanguageTranslationModel:
        """Create model from config."""
        encoder_type = self.config.get('encoder_type', 'stgcn')
        
        if encoder_type == 'stgcn':
            encoder = STGCN(
                num_joints=self.config.get('num_joints', 21),
                in_channels=self.config.get('in_channels', 3),
                hidden_channels=self.config.get('hidden_channels', 64),
                num_layers=self.config.get('encoder_layers', 6),
                output_dim=self.config.get('d_model', 256),
                dropout=0.0  # No dropout during inference
            )
        elif encoder_type == 'transformer':
            encoder = MotionTransformer(
                num_joints=self.config.get('num_joints', 21),
                in_channels=self.config.get('in_channels', 3),
                d_model=self.config.get('d_model', 256),
                num_heads=self.config.get('num_heads', 8),
                num_layers=self.config.get('encoder_layers', 6),
                dim_feedforward=self.config.get('dim_feedforward', 1024),
                dropout=0.0
            )
        elif encoder_type == 'hybrid':
            encoder = HybridEncoder(
                num_joints=self.config.get('num_joints', 21),
                in_channels=self.config.get('in_channels', 3),
                stgcn_channels=self.config.get('hidden_channels', 64),
                stgcn_layers=self.config.get('stgcn_layers', 4),
                d_model=self.config.get('d_model', 256),
                num_heads=self.config.get('num_heads', 8),
                transformer_layers=self.config.get('transformer_layers', 4),
                dropout=0.0
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        model = SignLanguageTranslationModel(
            encoder=encoder,
            vocab_size=self.config.get('vocab_size', 10000),
            d_model=self.config.get('d_model', 256),
            num_decoder_layers=self.config.get('decoder_layers', 4),
            num_heads=self.config.get('num_heads', 8),
            dim_feedforward=self.config.get('dim_feedforward', 1024),
            dropout=0.0,
            max_len=self.config.get('max_text_length', 512),
            pad_token_id=self.config.get('pad_token_id', 0),
            bos_token_id=self.config.get('bos_token_id', 1),
            eos_token_id=self.config.get('eos_token_id', 2)
        )
        
        return model.to(self.device)
    
    def preprocess(
        self,
        skeleton_data: Union[np.ndarray, List[List[Dict]], torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess skeleton data for inference.
        
        Args:
            skeleton_data: Skeleton data in various formats:
                - numpy array: (T, V, 3) or (B, T, V, 3)
                - list of frames: [[{joint, x, y, z}, ...], ...]
                - torch tensor
        
        Returns:
            Preprocessed tensor (B, T, V, 3)
        """
        # Convert to numpy if needed
        if isinstance(skeleton_data, list):
            skeleton_data = self._parse_frame_list(skeleton_data)
        elif isinstance(skeleton_data, torch.Tensor):
            skeleton_data = skeleton_data.numpy()
        
        # Ensure 4D
        if skeleton_data.ndim == 3:
            skeleton_data = skeleton_data[np.newaxis, ...]
        
        # Normalize
        skeleton_data = (skeleton_data - self.mean) / self.std
        
        # Convert to tensor
        tensor = torch.tensor(skeleton_data, dtype=torch.float32)
        
        return tensor.to(self.device)
    
    def _parse_frame_list(self, frames: List[List[Dict]]) -> np.ndarray:
        """Parse list of frame dictionaries to numpy array."""
        num_joints = self.config.get('num_joints', 21)
        num_frames = len(frames)
        
        skeleton = np.zeros((num_frames, num_joints, 3))
        
        for t, frame in enumerate(frames):
            for joint in frame:
                idx = joint.get('index', 0)
                if idx < num_joints:
                    skeleton[t, idx, 0] = joint.get('x', 0)
                    skeleton[t, idx, 1] = joint.get('y', 0)
                    skeleton[t, idx, 2] = joint.get('z', 0)
        
        return skeleton
    
    @torch.no_grad()
    def translate(
        self,
        skeleton_data: Union[np.ndarray, List, torch.Tensor],
        max_length: int = 100,
        beam_size: int = 5,
        return_scores: bool = False,
        return_alternatives: bool = False,
        num_alternatives: int = 5
    ) -> Union[str, Tuple[str, float], Dict]:
        """
        Translate skeleton sequence to text.
        
        Args:
            skeleton_data: Input skeleton data
            max_length: Maximum output length
            beam_size: Beam search size
            return_scores: Whether to return confidence scores
            return_alternatives: Whether to return alternative translations
            num_alternatives: Number of alternatives if requested
        
        Returns:
            Depending on options:
            - Just text string
            - Tuple of (text, score)
            - Dict with text, score, and alternatives
        """
        # Preprocess
        tensor = self.preprocess(skeleton_data)
        
        # Generate
        generated, scores = self.model.translate(
            tensor,
            max_length=max_length,
            beam_size=beam_size if not return_alternatives else num_alternatives
        )
        
        # Decode text
        if self.tokenizer is not None:
            text = self.tokenizer.decode(
                generated[0].tolist(),
                skip_special_tokens=True
            )
        else:
            # Character-level fallback
            text = ''.join([
                chr(c) for c in generated[0].tolist()
                if c > 2 and c < 128
            ])
        
        score = scores[0].item()
        
        # Return based on options
        if return_alternatives:
            alternatives = []
            for i in range(min(num_alternatives, generated.shape[0])):
                if self.tokenizer:
                    alt_text = self.tokenizer.decode(
                        generated[i].tolist(),
                        skip_special_tokens=True
                    )
                else:
                    alt_text = ''.join([
                        chr(c) for c in generated[i].tolist()
                        if c > 2 and c < 128
                    ])
                alternatives.append({
                    'text': alt_text,
                    'score': scores[i].item() if i < len(scores) else 0.0
                })
            
            return {
                'text': text,
                'score': score,
                'confidence': self._score_to_confidence(score),
                'alternatives': alternatives
            }
        
        if return_scores:
            return text, score
        
        return text
    
    def _score_to_confidence(self, log_score: float) -> float:
        """Convert log probability score to confidence [0, 1]."""
        # Normalize using sigmoid on scaled log score
        return 1.0 / (1.0 + np.exp(-log_score / 10.0))
    
    @torch.no_grad()
    def translate_batch(
        self,
        skeleton_batch: Union[np.ndarray, torch.Tensor],
        max_length: int = 100,
        beam_size: int = 5
    ) -> List[Dict]:
        """
        Translate a batch of skeleton sequences.
        
        Args:
            skeleton_batch: Batch of skeletons (B, T, V, 3)
            max_length: Maximum output length
            beam_size: Beam search size
        
        Returns:
            List of result dictionaries
        """
        # Preprocess
        if isinstance(skeleton_batch, np.ndarray):
            tensor = torch.tensor(skeleton_batch, dtype=torch.float32)
            tensor = (tensor - torch.tensor(self.mean)) / torch.tensor(self.std)
            tensor = tensor.to(self.device)
        else:
            tensor = skeleton_batch.to(self.device)
        
        # Generate
        generated, scores = self.model.translate(
            tensor,
            max_length=max_length,
            beam_size=beam_size
        )
        
        # Decode all
        results = []
        for i in range(generated.shape[0]):
            if self.tokenizer:
                text = self.tokenizer.decode(
                    generated[i].tolist(),
                    skip_special_tokens=True
                )
            else:
                text = ''.join([
                    chr(c) for c in generated[i].tolist()
                    if c > 2 and c < 128
                ])
            
            results.append({
                'text': text,
                'score': scores[i].item(),
                'confidence': self._score_to_confidence(scores[i].item())
            })
        
        return results
    
    def get_encoder_features(
        self,
        skeleton_data: Union[np.ndarray, List, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get encoder features without decoding.
        
        Useful for visualization or downstream tasks.
        
        Args:
            skeleton_data: Input skeleton data
        
        Returns:
            Encoder features (B, T', D)
        """
        tensor = self.preprocess(skeleton_data)
        
        with torch.no_grad():
            features = self.model.encoder(tensor, return_sequence=True)
        
        return features


class StreamingTranslator:
    """
    Streaming translator for real-time inference.
    
    Accumulates frames and runs inference when enough
    frames are collected or when explicitly triggered.
    """
    
    def __init__(
        self,
        translator: SignLanguageTranslator,
        window_size: int = 60,
        stride: int = 30,
        min_frames: int = 30
    ):
        """
        Initialize streaming translator.
        
        Args:
            translator: Base translator instance
            window_size: Number of frames per inference window
            stride: Number of frames to advance between inferences
            min_frames: Minimum frames before first inference
        """
        self.translator = translator
        self.window_size = window_size
        self.stride = stride
        self.min_frames = min_frames
        
        # Frame buffer
        self.frames: List[List[Dict]] = []
        self.last_inference_idx = 0
    
    def add_frame(self, frame: List[Dict]) -> Optional[Dict]:
        """
        Add a frame and optionally trigger inference.
        
        Args:
            frame: Single frame of joint data
        
        Returns:
            Inference result if triggered, else None
        """
        self.frames.append(frame)
        
        # Check if we should run inference
        if len(self.frames) >= self.min_frames:
            if len(self.frames) - self.last_inference_idx >= self.stride:
                return self.run_inference()
        
        return None
    
    def run_inference(self) -> Dict:
        """
        Run inference on current frame buffer.
        
        Returns:
            Translation result
        """
        # Get window
        start_idx = max(0, len(self.frames) - self.window_size)
        window = self.frames[start_idx:]
        
        # Translate
        result = self.translator.translate(
            window,
            return_alternatives=False,
            return_scores=True
        )
        
        self.last_inference_idx = len(self.frames)
        
        return {
            'text': result[0],
            'score': result[1],
            'frames_used': len(window),
            'total_frames': len(self.frames)
        }
    
    def reset(self):
        """Clear the frame buffer."""
        self.frames = []
        self.last_inference_idx = 0
    
    def finalize(self) -> Optional[Dict]:
        """
        Finalize and return any remaining translation.
        
        Call this when the gesture sequence ends.
        """
        if len(self.frames) >= self.min_frames:
            result = self.run_inference()
            self.reset()
            return result
        
        self.reset()
        return None


def load_translator(
    checkpoint_path: str,
    device: Optional[str] = None,
    **kwargs
) -> SignLanguageTranslator:
    """
    Convenience function to load a translator.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device string ('cuda', 'cpu', or None for auto)
        **kwargs: Additional arguments for translator
    
    Returns:
        SignLanguageTranslator instance
    """
    if device:
        device = torch.device(device)
    
    return SignLanguageTranslator(
        checkpoint_path=checkpoint_path,
        device=device,
        **kwargs
    )


if __name__ == '__main__':
    # Example usage
    print("Sign Language Translation Inference Module")
    print("=" * 50)
    
    # Create dummy checkpoint for testing
    import tempfile
    from models.stgcn import STGCN
    from models.decoder import SignLanguageTranslationModel
    
    # Create model
    config = {
        'encoder_type': 'stgcn',
        'num_joints': 21,
        'in_channels': 3,
        'd_model': 256,
        'hidden_channels': 64,
        'encoder_layers': 4,
        'decoder_layers': 4,
        'num_heads': 8,
        'dim_feedforward': 1024,
        'vocab_size': 1000,
        'max_text_length': 100,
        'pad_token_id': 0,
        'bos_token_id': 1,
        'eos_token_id': 2,
    }
    
    encoder = STGCN(
        num_joints=21,
        in_channels=3,
        hidden_channels=64,
        num_layers=4,
        output_dim=256
    )
    
    model = SignLanguageTranslationModel(
        encoder=encoder,
        vocab_size=1000,
        d_model=256,
        num_decoder_layers=4,
        num_heads=8
    )
    
    # Save dummy checkpoint
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config
        }
        torch.save(checkpoint, f.name)
        checkpoint_path = f.name
    
    print(f"Created test checkpoint: {checkpoint_path}")
    
    # Load translator
    translator = load_translator(checkpoint_path)
    
    # Test inference
    dummy_skeleton = np.random.randn(60, 21, 3)
    
    result = translator.translate(
        dummy_skeleton,
        beam_size=3,
        return_alternatives=True
    )
    
    print(f"\nTranslation result:")
    print(f"  Text: {result['text']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Alternatives: {len(result['alternatives'])}")
    
    # Test streaming
    print("\nTesting streaming translator...")
    streaming = StreamingTranslator(translator, window_size=30, stride=15)
    
    for i in range(60):
        frame = [
            {'joint': f'joint_{j}', 'x': 0.1, 'y': 0.2, 'z': 0.3, 'index': j}
            for j in range(21)
        ]
        result = streaming.add_frame(frame)
        if result:
            print(f"  Frame {i}: {result['text'][:20]}...")
    
    final = streaming.finalize()
    if final:
        print(f"  Final: {final['text'][:20]}...")
    
    # Cleanup
    import os
    os.unlink(checkpoint_path)
    
    print("\nInference module test complete!")
