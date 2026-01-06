"""
Transformer Decoder for Sign Language Translation

This module implements a Transformer-based decoder that converts
motion embeddings into text sequences (sign language translation).

The decoder uses:
- Cross-attention to attend to encoder outputs
- Autoregressive generation with causal masking
- BPE tokenization for subword units

Reference:
    Vaswani et al. (2017). Attention Is All You Need.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention with future masking.
    
    Prevents the decoder from attending to future tokens.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        max_len: int = 512
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(max_len, max_len, dtype=torch.bool),
            diagonal=1
        )
        self.register_buffer('causal_mask', causal_mask)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with causal masking.
        
        Args:
            x: Input tensor (B, T, D)
            padding_mask: Padding mask (B, T), True = padded
        
        Returns:
            Output tensor (B, T, D)
        """
        B, T, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T)
        
        # Apply causal mask
        causal_mask = self.causal_mask[:T, :T]
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply padding mask
        if padding_mask is not None:
            attn = attn.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, T, D_head)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """
    Cross-attention layer for attending to encoder outputs.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Decoder input (B, T_dec, D)
            encoder_output: Encoder output (B, T_enc, D)
            encoder_padding_mask: Encoder padding mask (B, T_enc)
        
        Returns:
            Output tensor (B, T_dec, D)
        """
        B, T_dec, D = x.shape
        T_enc = encoder_output.shape[1]
        
        # Project queries from decoder, keys/values from encoder
        q = self.q_proj(x).view(B, T_dec, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(encoder_output).view(B, T_enc, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_output).view(B, T_enc, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T_dec, T_enc)
        
        # Apply encoder padding mask
        if encoder_padding_mask is not None:
            attn = attn.masked_fill(
                encoder_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, H, T_dec, D_head)
        out = out.transpose(1, 2).contiguous().view(B, T_dec, D)
        
        return self.out_proj(out)


class DecoderLayer(nn.Module):
    """
    Single Transformer decoder layer.
    
    Consists of:
    1. Causal self-attention
    2. Cross-attention to encoder
    3. Feed-forward network
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.self_attn_norm = nn.LayerNorm(d_model)
        
        # Cross-attention
        self.cross_attn = CrossAttention(d_model, num_heads, dropout)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Decoder input (B, T_dec, D)
            encoder_output: Encoder output (B, T_enc, D)
            tgt_padding_mask: Target padding mask (B, T_dec)
            encoder_padding_mask: Encoder padding mask (B, T_enc)
        
        Returns:
            Output tensor (B, T_dec, D)
        """
        # Pre-norm architecture
        # Self-attention
        residual = x
        x = self.self_attn_norm(x)
        x = self.self_attn(x, tgt_padding_mask)
        x = residual + x
        
        # Cross-attention
        residual = x
        x = self.cross_attn_norm(x)
        x = self.cross_attn(x, encoder_output, encoder_padding_mask)
        x = residual + x
        
        # Feed-forward
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


class SignLanguageDecoder(nn.Module):
    """
    Transformer decoder for sign language translation.
    
    Converts motion embeddings from the encoder into text sequences.
    Supports autoregressive generation with beam search.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2
    ):
        """
        Initialize decoder.
        
        Args:
            vocab_size: Size of the vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
            dim_feedforward: Feed-forward network dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
            pad_token_id: Padding token ID
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        self.embed_dropout = nn.Dropout(dropout)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie output projection with input embeddings
        self.output_proj.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        tgt_tokens: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass (training mode).
        
        Args:
            tgt_tokens: Target token IDs (B, T_dec)
            encoder_output: Encoder output (B, T_enc, D)
            tgt_padding_mask: Target padding mask (B, T_dec)
            encoder_padding_mask: Encoder padding mask (B, T_enc)
        
        Returns:
            Logits (B, T_dec, vocab_size)
        """
        B, T = tgt_tokens.shape
        
        # Embed tokens
        x = self.token_embedding(tgt_tokens)  # (B, T, D)
        x = x + self.pos_embedding[:, :T, :]
        x = self.embed_dropout(x)
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_padding_mask, encoder_padding_mask)
        
        # Output projection
        x = self.output_norm(x)
        logits = self.output_proj(x)  # (B, T, vocab_size)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        encoder_output: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        beam_size: int = 5,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate text autoregressively.
        
        Args:
            encoder_output: Encoder output (B, T_enc, D)
            encoder_padding_mask: Encoder padding mask (B, T_enc)
            max_length: Maximum generation length
            beam_size: Beam search size (1 = greedy)
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
        
        Returns:
            Tuple of (generated tokens, scores)
        """
        B = encoder_output.shape[0]
        device = encoder_output.device
        
        if beam_size == 1:
            return self._greedy_generate(
                encoder_output, encoder_padding_mask, max_length,
                temperature, top_k, top_p
            )
        else:
            return self._beam_search(
                encoder_output, encoder_padding_mask, max_length, beam_size
            )
    
    def _greedy_generate(
        self,
        encoder_output: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor],
        max_length: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy/sampling generation."""
        B = encoder_output.shape[0]
        device = encoder_output.device
        
        # Start with BOS token
        generated = torch.full(
            (B, 1), self.bos_token_id, dtype=torch.long, device=device
        )
        scores = torch.zeros(B, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for _ in range(max_length - 1):
            # Get logits for last token
            logits = self.forward(
                generated, encoder_output,
                encoder_padding_mask=encoder_padding_mask
            )
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample or argmax
            probs = F.softmax(logits, dim=-1)
            if temperature == 0:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Update scores
            token_scores = torch.gather(
                F.log_softmax(logits, dim=-1), 1, next_token
            ).squeeze(-1)
            scores = scores + token_scores * (~finished).float()
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            finished = finished | (next_token.squeeze(-1) == self.eos_token_id)
            if finished.all():
                break
        
        return generated, scores
    
    def _beam_search(
        self,
        encoder_output: torch.Tensor,
        encoder_padding_mask: Optional[torch.Tensor],
        max_length: int,
        beam_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Beam search decoding."""
        B = encoder_output.shape[0]
        device = encoder_output.device
        
        # Expand encoder output for beam search
        encoder_output = encoder_output.unsqueeze(1).expand(-1, beam_size, -1, -1)
        encoder_output = encoder_output.contiguous().view(B * beam_size, -1, encoder_output.shape[-1])
        
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.unsqueeze(1).expand(-1, beam_size, -1)
            encoder_padding_mask = encoder_padding_mask.contiguous().view(B * beam_size, -1)
        
        # Initialize beams
        generated = torch.full(
            (B * beam_size, 1), self.bos_token_id, dtype=torch.long, device=device
        )
        beam_scores = torch.zeros(B * beam_size, device=device)
        beam_scores[1::beam_size] = float('-inf')  # Only first beam is valid initially
        
        finished_beams = []
        finished_scores = []
        
        for step in range(max_length - 1):
            # Get logits
            logits = self.forward(
                generated, encoder_output,
                encoder_padding_mask=encoder_padding_mask
            )
            logits = logits[:, -1, :]  # (B * beam_size, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Calculate scores for all possible next tokens
            vocab_size = log_probs.shape[-1]
            next_scores = beam_scores.unsqueeze(-1) + log_probs  # (B * beam_size, vocab_size)
            next_scores = next_scores.view(B, beam_size * vocab_size)
            
            # Get top-k candidates
            top_scores, top_indices = torch.topk(next_scores, beam_size, dim=-1)
            
            # Calculate beam and token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Update beams
            batch_indices = torch.arange(B, device=device).unsqueeze(-1).expand(-1, beam_size)
            prev_beam_indices = batch_indices * beam_size + beam_indices
            
            generated = torch.cat([
                generated[prev_beam_indices.view(-1)],
                token_indices.view(-1, 1)
            ], dim=1)
            
            beam_scores = top_scores.view(-1)
            
            # Check for finished beams (EOS token)
            eos_mask = token_indices == self.eos_token_id
            
            if eos_mask.any():
                for b in range(B):
                    for k in range(beam_size):
                        if eos_mask[b, k]:
                            idx = b * beam_size + k
                            if len(finished_beams) <= b:
                                finished_beams.append([])
                                finished_scores.append([])
                            if len(finished_beams[b]) < beam_size:
                                finished_beams[b].append(generated[idx].clone())
                                finished_scores[b].append(beam_scores[idx].item())
        
        # Return best beams
        best_seqs = []
        best_scores = []
        
        for b in range(B):
            if b < len(finished_beams) and finished_beams[b]:
                best_idx = max(range(len(finished_scores[b])), key=lambda i: finished_scores[b][i])
                best_seqs.append(finished_beams[b][best_idx])
                best_scores.append(finished_scores[b][best_idx])
            else:
                # No finished beam, take best current beam
                idx = b * beam_size
                best_seqs.append(generated[idx])
                best_scores.append(beam_scores[idx].item())
        
        # Pad sequences to same length
        max_len = max(seq.shape[0] for seq in best_seqs)
        padded_seqs = torch.full(
            (B, max_len), self.pad_token_id, dtype=torch.long, device=device
        )
        for b, seq in enumerate(best_seqs):
            padded_seqs[b, :seq.shape[0]] = seq
        
        return padded_seqs, torch.tensor(best_scores, device=device)


class SignLanguageTranslationModel(nn.Module):
    """
    Complete Sign Language Translation model.
    
    Combines encoder (ST-GCN or Motion Transformer) with decoder
    for end-to-end translation from skeleton sequences to text.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        vocab_size: int,
        d_model: int = 256,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2
    ):
        """
        Initialize translation model.
        
        Args:
            encoder: Skeleton encoder (STGCN or MotionTransformer)
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            dim_feedforward: Feed-forward dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
            pad_token_id: Padding token ID
            bos_token_id: BOS token ID
            eos_token_id: EOS token ID
        """
        super().__init__()
        
        self.encoder = encoder
        
        self.decoder = SignLanguageDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )
        
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
    
    def forward(
        self,
        skeleton_seq: torch.Tensor,
        tgt_tokens: torch.Tensor,
        skeleton_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass (training).
        
        Args:
            skeleton_seq: Skeleton sequence (B, T, V, C)
            tgt_tokens: Target token IDs (B, T_tgt)
            skeleton_padding_mask: Skeleton padding mask (B, T)
            tgt_padding_mask: Target padding mask (B, T_tgt)
        
        Returns:
            Logits (B, T_tgt, vocab_size)
        """
        # Encode skeleton
        encoder_output = self.encoder(skeleton_seq, return_sequence=True)
        
        # Decode to text
        logits = self.decoder(
            tgt_tokens,
            encoder_output,
            tgt_padding_mask=tgt_padding_mask,
            encoder_padding_mask=skeleton_padding_mask
        )
        
        return logits
    
    @torch.no_grad()
    def translate(
        self,
        skeleton_seq: torch.Tensor,
        skeleton_padding_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        beam_size: int = 5,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Translate skeleton sequence to text.
        
        Args:
            skeleton_seq: Skeleton sequence (B, T, V, C)
            skeleton_padding_mask: Padding mask
            max_length: Maximum output length
            beam_size: Beam search size
        
        Returns:
            Tuple of (generated token IDs, scores)
        """
        # Encode skeleton
        encoder_output = self.encoder(skeleton_seq, return_sequence=True)
        
        # Generate text
        return self.decoder.generate(
            encoder_output,
            encoder_padding_mask=skeleton_padding_mask,
            max_length=max_length,
            beam_size=beam_size,
            **kwargs
        )


if __name__ == '__main__':
    from .stgcn import STGCN
    
    # Test decoder
    batch_size = 4
    num_frames = 60
    num_joints = 21
    in_channels = 3
    vocab_size = 1000
    d_model = 256
    
    # Create encoder
    encoder = STGCN(
        num_joints=num_joints,
        in_channels=in_channels,
        hidden_channels=64,
        num_layers=4,
        output_dim=d_model
    )
    
    # Create translation model
    model = SignLanguageTranslationModel(
        encoder=encoder,
        vocab_size=vocab_size,
        d_model=d_model,
        num_decoder_layers=4,
        num_heads=8
    )
    
    # Random inputs
    skeleton_seq = torch.randn(batch_size, num_frames, num_joints, in_channels)
    tgt_tokens = torch.randint(0, vocab_size, (batch_size, 20))
    
    # Forward pass (training)
    logits = model(skeleton_seq, tgt_tokens)
    print(f"Logits shape: {logits.shape}")  # (B, T_tgt, vocab_size)
    
    # Generation (inference)
    model.eval()
    generated, scores = model.translate(skeleton_seq, beam_size=3, max_length=30)
    print(f"Generated shape: {generated.shape}")
    print(f"Scores shape: {scores.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
