"""
Motion Transformer for Sign Language Encoding

This module implements a Transformer-based encoder for 3D skeletal motion.
The Motion Transformer directly processes joint sequences without the
graph structure, relying on self-attention to learn joint relationships.

Advantages over ST-GCN:
- Better at capturing long-range temporal dependencies
- More flexible joint relationship modeling
- Easier to scale to larger models

Reference:
    Inspired by ViT (Vision Transformer) architecture adapted for sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequences.
    
    Adds position information to the input embeddings.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 500,
        dropout: float = 0.1
    ):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (B, T, D)
        
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding.
    
    Uses trainable embeddings instead of fixed sinusoidal patterns.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 500,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embedding[:, :x.size(1)]
        return self.dropout(x)


class JointEmbedding(nn.Module):
    """
    Embeds per-joint coordinates into a higher dimensional space.
    
    Projects (x, y, z) coordinates to d_model dimensions with
    optional joint-specific embeddings.
    """
    
    def __init__(
        self,
        num_joints: int,
        in_channels: int = 3,
        d_model: int = 256,
        use_joint_embedding: bool = True
    ):
        """
        Initialize joint embedding.
        
        Args:
            num_joints: Number of skeleton joints
            in_channels: Input channels per joint (3 for x, y, z)
            d_model: Output embedding dimension
            use_joint_embedding: Whether to add learnable joint type embeddings
        """
        super().__init__()
        
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.d_model = d_model
        
        # Linear projection for coordinates
        self.coord_proj = nn.Linear(in_channels, d_model)
        
        # Optional joint type embedding
        if use_joint_embedding:
            self.joint_embedding = nn.Parameter(
                torch.randn(1, 1, num_joints, d_model) * 0.02
            )
        else:
            self.joint_embedding = None
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed joint coordinates.
        
        Args:
            x: Input tensor (B, T, V, C) or (B, T, V*C)
        
        Returns:
            Embedded tensor (B, T, V, D) or (B, T*V, D)
        """
        B, T = x.shape[:2]
        
        # Ensure correct shape
        if x.dim() == 3:
            # (B, T, V*C) -> (B, T, V, C)
            x = x.view(B, T, self.num_joints, self.in_channels)
        
        # Project coordinates
        x = self.coord_proj(x)  # (B, T, V, D)
        
        # Add joint embeddings
        if self.joint_embedding is not None:
            x = x + self.joint_embedding
        
        x = self.norm(x)
        
        return x


class MotionTransformerEncoder(nn.Module):
    """
    Transformer encoder for motion sequences.
    
    Processes joint sequences with self-attention to capture
    both spatial (inter-joint) and temporal (inter-frame) relationships.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize transformer encoder.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()
        
        self.d_model = d_model
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-LayerNorm for better training
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, T, D)
            mask: Attention mask (T, T)
            src_key_padding_mask: Padding mask (B, T)
        
        Returns:
            Encoded tensor (B, T, D)
        """
        return self.encoder(
            x,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask
        )


class MotionTransformer(nn.Module):
    """
    Complete Motion Transformer for skeleton sequence encoding.
    
    Combines joint embedding, positional encoding, and transformer
    encoder to produce motion representations for sign language.
    """
    
    def __init__(
        self,
        num_joints: int = 21,
        in_channels: int = 3,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 500,
        use_joint_tokens: bool = True,
        pool_joints: bool = True
    ):
        """
        Initialize Motion Transformer.
        
        Args:
            num_joints: Number of skeleton joints
            in_channels: Input channels per joint (3 for x, y, z)
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
            use_joint_tokens: If True, each joint is a token;
                            if False, all joints at each timestep are one token
            pool_joints: If True and use_joint_tokens, pool joints per frame
        """
        super().__init__()
        
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.d_model = d_model
        self.use_joint_tokens = use_joint_tokens
        self.pool_joints = pool_joints
        
        # Joint embedding
        if use_joint_tokens:
            self.joint_embed = JointEmbedding(
                num_joints, in_channels, d_model, use_joint_embedding=True
            )
            # Temporal positional encoding
            self.temporal_pos = LearnedPositionalEncoding(d_model, max_len, dropout)
            # Spatial (joint) positional encoding is in JointEmbedding
        else:
            # Flatten all joints into single token per frame
            self.joint_embed = nn.Sequential(
                nn.Linear(num_joints * in_channels, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
            )
            self.temporal_pos = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer encoder
        self.encoder = MotionTransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # CLS token for sequence-level representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = True
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input skeleton sequence (B, T, V, C) or (B, T, V*C)
            padding_mask: Mask for padded frames (B, T), True = padded
            return_sequence: If True, return per-frame features;
                           if False, return CLS token embedding
        
        Returns:
            If return_sequence: (B, T, D)
            If not: (B, D)
        """
        B, T = x.shape[:2]
        
        if self.use_joint_tokens:
            # Each joint becomes a token
            x = self.joint_embed(x)  # (B, T, V, D)
            
            # Add temporal position encoding (broadcast over joints)
            # Create temporal positions and add to joint embeddings
            temporal_positions = self.temporal_pos.pos_embedding[:, :T, :]  # (1, T, D)
            x = x + temporal_positions.unsqueeze(2)  # (B, T, V, D)
            
            if self.pool_joints:
                # Pool joints to get per-frame representation
                x = x.mean(dim=2)  # (B, T, D)
            else:
                # Flatten time and joints
                x = x.view(B, T * self.num_joints, self.d_model)
        else:
            # All joints as single token per frame
            x = x.view(B, T, -1)  # (B, T, V*C)
            x = self.joint_embed(x)  # (B, T, D)
            x = self.temporal_pos(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+T, D)
        
        # Update padding mask for CLS token
        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        
        # Apply transformer encoder
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.output_norm(x)
        
        if return_sequence:
            # Return sequence (excluding CLS token)
            return x[:, 1:, :]
        else:
            # Return CLS token embedding
            return x[:, 0, :]
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Get attention weights for visualization.
        
        Useful for interpretability analysis.
        """
        # This would require modifying the encoder to store attention weights
        # Implementation left as TODO for research purposes
        raise NotImplementedError("Attention weight extraction not yet implemented")


class HybridEncoder(nn.Module):
    """
    Hybrid encoder combining ST-GCN and Transformer.
    
    Uses ST-GCN for local spatial-temporal modeling,
    followed by Transformer for global sequence modeling.
    """
    
    def __init__(
        self,
        num_joints: int = 21,
        in_channels: int = 3,
        stgcn_channels: int = 64,
        stgcn_layers: int = 4,
        d_model: int = 256,
        num_heads: int = 8,
        transformer_layers: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize hybrid encoder.
        
        Args:
            num_joints: Number of skeleton joints
            in_channels: Input channels per joint
            stgcn_channels: Hidden channels for ST-GCN
            stgcn_layers: Number of ST-GCN layers
            d_model: Transformer model dimension
            num_heads: Number of attention heads
            transformer_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        
        from .stgcn import STGCN
        
        # ST-GCN for local modeling
        self.stgcn = STGCN(
            num_joints=num_joints,
            in_channels=in_channels,
            hidden_channels=stgcn_channels,
            num_layers=stgcn_layers,
            output_dim=d_model,
            dropout=dropout
        )
        
        # Positional encoding for transformer
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer for global modeling
        self.transformer = MotionTransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=transformer_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout
        )
        
        self.output_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_sequence: bool = True
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input skeleton sequence (B, T, V, C)
            padding_mask: Mask for padded frames (B, T)
            return_sequence: Whether to return sequence or pooled embedding
        
        Returns:
            Encoded features
        """
        # ST-GCN encoding
        x = self.stgcn(x, return_sequence=True)  # (B, T', D)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.output_norm(x)
        
        if return_sequence:
            return x
        else:
            # Global average pooling
            if padding_mask is not None:
                # Mask out padded positions
                mask = ~padding_mask.unsqueeze(-1)
                x = (x * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                x = x.mean(dim=1)
            return x


if __name__ == '__main__':
    # Test Motion Transformer
    batch_size = 4
    num_frames = 60
    num_joints = 21
    in_channels = 3
    d_model = 256
    
    model = MotionTransformer(
        num_joints=num_joints,
        in_channels=in_channels,
        d_model=d_model,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    )
    
    # Random input: (B, T, V, C)
    x = torch.randn(batch_size, num_frames, num_joints, in_channels)
    
    # Test sequence output
    out_seq = model(x, return_sequence=True)
    print(f"Input shape: {x.shape}")
    print(f"Sequence output shape: {out_seq.shape}")
    
    # Test CLS token output
    out_cls = model(x, return_sequence=False)
    print(f"CLS output shape: {out_cls.shape}")
    
    # Test with padding mask
    padding_mask = torch.zeros(batch_size, num_frames, dtype=torch.bool)
    padding_mask[:, -10:] = True  # Last 10 frames are padded
    
    out_masked = model(x, padding_mask=padding_mask, return_sequence=True)
    print(f"Masked output shape: {out_masked.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
