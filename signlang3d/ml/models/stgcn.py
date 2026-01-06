"""
Spatial-Temporal Graph Convolutional Network (ST-GCN) for Sign Language

This module implements ST-GCN for encoding 3D skeletal motion data.
ST-GCN is designed for skeleton-based action recognition and is
particularly suited for sign language gesture encoding.

Key concepts:
- Skeleton as a graph: joints are nodes, bones are edges
- Spatial convolution: captures relationships between connected joints
- Temporal convolution: captures motion patterns over time

Reference:
    Yan, S., Xiong, Y., & Lin, D. (2018).
    Spatial Temporal Graph Convolutional Networks for Skeleton-Based
    Action Recognition. AAAI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class Graph:
    """
    Skeleton graph structure for ST-GCN.
    
    Defines the adjacency matrix based on skeleton topology.
    Supports different partitioning strategies for the graph convolution.
    """
    
    # Hand skeleton with 21 joints (MediaPipe standard)
    HAND_EDGES = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index finger
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]
    
    # Two-hand skeleton (42 joints)
    TWO_HAND_EDGES = (
        HAND_EDGES + 
        [(i + 21, j + 21) for i, j in HAND_EDGES] +
        [(0, 21)]  # Connect wrists
    )
    
    def __init__(
        self,
        num_joints: int = 21,
        edges: Optional[list] = None,
        partition_strategy: str = 'spatial'
    ):
        """
        Initialize skeleton graph.
        
        Args:
            num_joints: Number of joints in skeleton
            edges: List of (i, j) tuples defining bone connections
            partition_strategy: 'uniform', 'spatial', or 'distance'
        """
        self.num_joints = num_joints
        self.edges = edges or self.HAND_EDGES
        self.partition_strategy = partition_strategy
        
        self.adjacency_matrix = self._build_adjacency_matrix()
        self.normalized_adjacency = self._normalize_adjacency()
    
    def _build_adjacency_matrix(self) -> torch.Tensor:
        """Build the basic adjacency matrix from edges."""
        A = torch.zeros(self.num_joints, self.num_joints)
        
        for i, j in self.edges:
            if i < self.num_joints and j < self.num_joints:
                A[i, j] = 1
                A[j, i] = 1  # Undirected graph
        
        # Add self-loops
        A = A + torch.eye(self.num_joints)
        
        return A
    
    def _normalize_adjacency(self) -> torch.Tensor:
        """
        Normalize adjacency matrix.
        
        Uses symmetric normalization: D^(-1/2) * A * D^(-1/2)
        """
        A = self.adjacency_matrix
        D = torch.sum(A, dim=0)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0
        D_inv_sqrt = torch.diag(D_inv_sqrt)
        
        return D_inv_sqrt @ A @ D_inv_sqrt
    
    def get_partitioned_adjacency(self) -> torch.Tensor:
        """
        Get partitioned adjacency matrices for spatial convolution.
        
        Returns:
            Tensor of shape (num_partitions, num_joints, num_joints)
        """
        if self.partition_strategy == 'uniform':
            # Single partition (all joints treated equally)
            return self.normalized_adjacency.unsqueeze(0)
        
        elif self.partition_strategy == 'spatial':
            # Three partitions: self, neighbors closer to root, neighbors farther
            A = self.adjacency_matrix
            
            # Self connections
            A_self = torch.eye(self.num_joints)
            
            # Neighbor connections (excluding self)
            A_neighbor = A - torch.eye(self.num_joints)
            
            # Stack partitions
            partitions = torch.stack([A_self, A_neighbor])
            
            # Normalize each partition
            for i in range(partitions.shape[0]):
                D = torch.sum(partitions[i], dim=0)
                D_inv = torch.pow(D, -1)
                D_inv[torch.isinf(D_inv)] = 0
                partitions[i] = torch.diag(D_inv) @ partitions[i]
            
            return partitions
        
        else:
            return self.normalized_adjacency.unsqueeze(0)


class SpatialGraphConvolution(nn.Module):
    """
    Spatial graph convolution layer.
    
    Applies convolution over the skeleton graph structure.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        use_attention: bool = True
    ):
        """
        Initialize spatial graph convolution.
        
        Args:
            in_channels: Number of input channels per joint
            out_channels: Number of output channels per joint
            adjacency: Partitioned adjacency tensor (K, V, V)
            use_attention: Whether to use adaptive attention
        """
        super().__init__()
        
        self.num_partitions = adjacency.shape[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Register adjacency as buffer (not a parameter)
        self.register_buffer('adjacency', adjacency)
        
        # Convolution weights for each partition
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * self.num_partitions,
            kernel_size=1
        )
        
        # Adaptive attention (learnable adjacency modifier)
        if use_attention:
            num_joints = adjacency.shape[1]
            self.attention = nn.Parameter(
                torch.zeros(self.num_partitions, num_joints, num_joints)
            )
        else:
            self.attention = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, T, V) where
               B = batch size
               C = channels
               T = temporal length
               V = number of joints
        
        Returns:
            Output tensor (B, C', T, V)
        """
        B, C, T, V = x.shape
        
        # Apply convolution
        x = self.conv(x)  # (B, C' * K, T, V)
        
        # Reshape for partitions
        x = x.view(B, self.num_partitions, self.out_channels, T, V)
        
        # Apply graph convolution with adjacency matrix
        adjacency = self.adjacency
        if self.attention is not None:
            adjacency = adjacency + self.attention
        
        # Matrix multiply with adjacency for each partition
        # x: (B, K, C', T, V) @ A: (K, V, V) -> (B, K, C', T, V)
        x = torch.einsum('bkctv,kvw->bkctw', x, adjacency)
        
        # Sum over partitions
        x = x.sum(dim=1)  # (B, C', T, V)
        
        return x


class TemporalConvolution(nn.Module):
    """
    Temporal convolution block.
    
    Applies 1D convolution along the temporal dimension.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0
    ):
        """
        Initialize temporal convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Temporal kernel size
            stride: Temporal stride
            dilation: Temporal dilation
            dropout: Dropout probability
        """
        super().__init__()
        
        # Calculate padding for 'same' convolution
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
                dilation=(dilation, 1)
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, T, V)
        
        Returns:
            Output tensor (B, C', T', V)
        """
        return self.conv(x)


class STGCNBlock(nn.Module):
    """
    Spatial-Temporal Graph Convolution Block.
    
    Combines spatial graph convolution with temporal convolution,
    with residual connection.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True
    ):
        """
        Initialize ST-GCN block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            adjacency: Partitioned adjacency tensor
            stride: Temporal stride (for downsampling)
            dropout: Dropout probability
            residual: Whether to use residual connection
        """
        super().__init__()
        
        self.spatial_conv = SpatialGraphConvolution(
            in_channels, out_channels, adjacency
        )
        
        self.temporal_conv = TemporalConvolution(
            out_channels, out_channels,
            kernel_size=9,
            stride=stride,
            dropout=dropout
        )
        
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection
        self.residual = residual
        if residual:
            if in_channels != out_channels or stride != 1:
                self.residual_conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, T, V)
        
        Returns:
            Output tensor (B, C', T', V)
        """
        res = self.residual_conv(x) if self.residual else 0
        
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        
        if self.residual:
            x = x + res
        
        return self.relu(x)


class STGCN(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network for skeleton encoding.
    
    Encodes 3D skeletal motion sequences into feature vectors
    suitable for sign language translation.
    """
    
    def __init__(
        self,
        num_joints: int = 21,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_layers: int = 6,
        output_dim: int = 256,
        dropout: float = 0.1,
        edges: Optional[list] = None
    ):
        """
        Initialize ST-GCN encoder.
        
        Args:
            num_joints: Number of skeleton joints
            in_channels: Input channels per joint (3 for x, y, z)
            hidden_channels: Hidden dimension
            num_layers: Number of ST-GCN blocks
            output_dim: Output embedding dimension
            dropout: Dropout probability
            edges: Custom skeleton edges (uses hand skeleton if None)
        """
        super().__init__()
        
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        
        # Build skeleton graph
        graph = Graph(num_joints=num_joints, edges=edges)
        adjacency = graph.get_partitioned_adjacency()
        
        # Data normalization
        self.data_bn = nn.BatchNorm1d(num_joints * in_channels)
        
        # Build network layers
        self.layers = nn.ModuleList()
        
        # Layer configuration: (out_channels, stride)
        layer_configs = self._get_layer_configs(num_layers, hidden_channels)
        
        current_channels = in_channels
        for i, (out_channels, stride) in enumerate(layer_configs):
            self.layers.append(
                STGCNBlock(
                    current_channels,
                    out_channels,
                    adjacency,
                    stride=stride,
                    dropout=dropout
                )
            )
            current_channels = out_channels
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(current_channels, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def _get_layer_configs(
        self,
        num_layers: int,
        hidden_channels: int
    ) -> list:
        """Generate layer configurations."""
        configs = []
        
        # First layer: expand from input channels
        configs.append((hidden_channels, 1))
        
        # Middle layers
        for i in range(1, num_layers - 1):
            # Double channels every 2 layers, stride 2 every 3 layers
            out_ch = hidden_channels * (2 ** (i // 2))
            out_ch = min(out_ch, hidden_channels * 4)  # Cap at 4x
            stride = 2 if i % 3 == 2 else 1
            configs.append((out_ch, stride))
        
        # Final layer
        if num_layers > 1:
            configs.append((hidden_channels * 4, 1))
        
        return configs
    
    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = True
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input skeleton sequence (B, T, V, C) or (B, C, T, V)
            return_sequence: If True, return per-frame features;
                           if False, return single sequence embedding
        
        Returns:
            If return_sequence: (B, T', D) where T' <= T
            If not: (B, D)
        """
        # Handle input shape
        if x.dim() == 4 and x.shape[-1] == self.in_channels:
            # (B, T, V, C) -> (B, C, T, V)
            x = x.permute(0, 3, 1, 2)
        
        B, C, T, V = x.shape
        
        # Normalize input
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, T, V, C)
        x = x.view(B * T, V * C)
        x = self.data_bn(x)
        x = x.view(B, T, V, C).permute(0, 3, 1, 2)  # (B, C, T, V)
        
        # Apply ST-GCN blocks
        for layer in self.layers:
            x = layer(x)
        
        if return_sequence:
            # Return per-frame features
            B, C, T_out, V = x.shape
            
            # Average over joints
            x = x.mean(dim=-1)  # (B, C, T_out)
            x = x.permute(0, 2, 1)  # (B, T_out, C)
            
            # Project to output dimension
            x = nn.functional.linear(
                x, 
                self.output_proj[2].weight, 
                self.output_proj[2].bias
            )
            
            return x
        else:
            # Return single embedding
            return self.output_proj(x)
    
    def get_output_length(self, input_length: int) -> int:
        """
        Calculate output sequence length given input length.
        
        Useful for planning decoder input size.
        """
        length = input_length
        for layer in self.layers:
            if hasattr(layer.temporal_conv.conv[2], 'stride'):
                stride = layer.temporal_conv.conv[2].stride[0]
                length = (length + stride - 1) // stride
        return length


if __name__ == '__main__':
    # Test the ST-GCN model
    batch_size = 4
    num_frames = 60
    num_joints = 21
    in_channels = 3
    
    model = STGCN(
        num_joints=num_joints,
        in_channels=in_channels,
        hidden_channels=64,
        num_layers=6,
        output_dim=256,
        dropout=0.1
    )
    
    # Random input: (B, T, V, C)
    x = torch.randn(batch_size, num_frames, num_joints, in_channels)
    
    # Test sequence output
    out_seq = model(x, return_sequence=True)
    print(f"Input shape: {x.shape}")
    print(f"Sequence output shape: {out_seq.shape}")
    
    # Test single embedding output
    out_embed = model(x, return_sequence=False)
    print(f"Embedding output shape: {out_embed.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
