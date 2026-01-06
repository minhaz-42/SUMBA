"""
Training Pipeline for Sign Language Translation

This module provides a complete training pipeline with:
- Mixed precision training
- Learning rate scheduling
- Checkpoint management
- Metric logging (BLEU, WER)
- Early stopping
- TensorBoard integration

Usage:
    python train.py --config config.yaml
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.stgcn import STGCN
from models.motion_transformer import MotionTransformer, HybridEncoder
from models.decoder import SignLanguageTranslationModel
from datasets.sign_language import SignLanguageDataset, SignLanguageCollator, create_dataloaders

# Metrics
try:
    from sacrebleu.metrics import BLEU
    from jiwer import wer as compute_wer
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("Warning: sacrebleu or jiwer not installed. Metrics will be limited.")


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for sign language translation models.
    
    Handles the complete training loop with validation,
    checkpointing, and metric computation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        tokenizer: Any,
        config: Dict[str, Any],
        checkpoint_dir: Path,
        log_dir: Path
    ):
        """
        Initialize trainer.
        
        Args:
            model: Translation model
            train_loader: Training data loader
            val_loader: Validation data loader
            tokenizer: Text tokenizer
            config: Training configuration
            checkpoint_dir: Directory for saving checkpoints
            log_dir: Directory for TensorBoard logs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config
        
        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=config.get('pad_token_id', 0),
            label_smoothing=config.get('label_smoothing', 0.1)
        )
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_bleu = 0.0
        self.patience_counter = 0
        
        # Gradient clipping
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Model parameters: {self._count_parameters():,}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with weight decay."""
        # Separate parameters with and without weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': self.config.get('weight_decay', 1e-5)
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0
            }
        ]
        
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        lr = self.config.get('learning_rate', 1e-4)
        
        if optimizer_name == 'adamw':
            return optim.AdamW(optimizer_grouped_parameters, lr=lr, betas=(0.9, 0.999))
        elif optimizer_name == 'adam':
            return optim.Adam(optimizer_grouped_parameters, lr=lr)
        elif optimizer_name == 'sgd':
            return optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_name = self.config.get('scheduler', 'cosine').lower()
        max_epochs = self.config.get('max_epochs', 100)
        warmup_epochs = self.config.get('warmup_epochs', 5)
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max_epochs - warmup_epochs,
                eta_min=1e-7
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        else:
            return None
    
    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}')
        
        for batch in pbar:
            # Move to device
            skeleton = batch['skeleton'].to(self.device)
            skeleton_mask = ~batch['skeleton_mask'].to(self.device)  # Invert for padding
            text_ids = batch['text_ids'].to(self.device)
            text_mask = ~batch['text_mask'].to(self.device)
            
            # Prepare decoder input/target (teacher forcing)
            tgt_input = text_ids[:, :-1]
            tgt_output = text_ids[:, 1:]
            tgt_mask = text_mask[:, :-1]
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    logits = self.model(
                        skeleton, tgt_input,
                        skeleton_padding_mask=skeleton_mask,
                        tgt_padding_mask=tgt_mask
                    )
                    
                    loss = self.criterion(
                        logits.contiguous().view(-1, logits.size(-1)),
                        tgt_output.contiguous().view(-1)
                    )
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(
                    skeleton, tgt_input,
                    skeleton_padding_mask=skeleton_mask,
                    tgt_padding_mask=tgt_mask
                )
                
                loss = self.criterion(
                    logits.contiguous().view(-1, logits.size(-1)),
                    tgt_output.contiguous().view(-1)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to TensorBoard
            if self.global_step % 100 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar(
                    'train/lr', self.optimizer.param_groups[0]['lr'], self.global_step
                )
        
        avg_loss = total_loss / num_batches
        
        return {'loss': avg_loss}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_references = []
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            # Move to device
            skeleton = batch['skeleton'].to(self.device)
            skeleton_mask = ~batch['skeleton_mask'].to(self.device)
            text_ids = batch['text_ids'].to(self.device)
            text_mask = ~batch['text_mask'].to(self.device)
            
            # Prepare decoder input/target
            tgt_input = text_ids[:, :-1]
            tgt_output = text_ids[:, 1:]
            tgt_mask = text_mask[:, :-1]
            
            # Forward pass
            if self.use_amp:
                with autocast():
                    logits = self.model(
                        skeleton, tgt_input,
                        skeleton_padding_mask=skeleton_mask,
                        tgt_padding_mask=tgt_mask
                    )
                    
                    loss = self.criterion(
                        logits.contiguous().view(-1, logits.size(-1)),
                        tgt_output.contiguous().view(-1)
                    )
            else:
                logits = self.model(
                    skeleton, tgt_input,
                    skeleton_padding_mask=skeleton_mask,
                    tgt_padding_mask=tgt_mask
                )
                
                loss = self.criterion(
                    logits.contiguous().view(-1, logits.size(-1)),
                    tgt_output.contiguous().view(-1)
                )
            
            total_loss += loss.item()
            num_batches += 1
            
            # Generate predictions for BLEU calculation (sample every N batches)
            if num_batches % 10 == 0:
                generated, _ = self.model.translate(
                    skeleton,
                    skeleton_padding_mask=skeleton_mask,
                    max_length=50,
                    beam_size=1
                )
                
                # Decode predictions and references
                for gen, ref in zip(generated, batch['transcripts']):
                    if self.tokenizer is not None:
                        pred_text = self.tokenizer.decode(gen.tolist(), skip_special_tokens=True)
                    else:
                        pred_text = ''.join([chr(c) for c in gen.tolist() if c > 0])
                    
                    all_predictions.append(pred_text)
                    all_references.append(ref)
        
        avg_loss = total_loss / num_batches
        
        metrics = {'val_loss': avg_loss}
        
        # Compute BLEU and WER
        if METRICS_AVAILABLE and all_predictions:
            try:
                bleu = BLEU()
                bleu_score = bleu.corpus_score(
                    all_predictions,
                    [all_references]
                ).score
                metrics['bleu'] = bleu_score
                
                # Compute WER
                wer_score = compute_wer(all_references, all_predictions)
                metrics['wer'] = wer_score
            except Exception as e:
                logger.warning(f"Error computing metrics: {e}")
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False, metrics: Dict = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_bleu': self.best_bleu,
            'config': self.config,
            'metrics': metrics,
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch:03d}.pt'
        torch.save(checkpoint, epoch_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint at epoch {self.epoch}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_bleu = checkpoint.get('best_bleu', 0.0)
        
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self):
        """Run the complete training loop."""
        max_epochs = self.config.get('max_epochs', 100)
        patience = self.config.get('early_stopping_patience', 10)
        warmup_epochs = self.config.get('warmup_epochs', 5)
        
        logger.info(f"Starting training for {max_epochs} epochs")
        
        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{max_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Val Loss: {val_metrics['val_loss']:.4f}"
            )
            
            if 'bleu' in val_metrics:
                logger.info(f"  BLEU: {val_metrics['bleu']:.2f}")
            if 'wer' in val_metrics:
                logger.info(f"  WER: {val_metrics['wer']:.4f}")
            
            # TensorBoard logging
            self.writer.add_scalar('val/loss', val_metrics['val_loss'], epoch)
            if 'bleu' in val_metrics:
                self.writer.add_scalar('val/bleu', val_metrics['bleu'], epoch)
            if 'wer' in val_metrics:
                self.writer.add_scalar('val/wer', val_metrics['wer'], epoch)
            
            # Update learning rate
            if self.scheduler:
                if epoch >= warmup_epochs:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step()
            
            # Check for improvement
            is_best = False
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                is_best = True
            else:
                self.patience_counter += 1
            
            # Update best BLEU
            if val_metrics.get('bleu', 0) > self.best_bleu:
                self.best_bleu = val_metrics['bleu']
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best, metrics={**train_metrics, **val_metrics})
            
            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        self.writer.close()
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Best BLEU score: {self.best_bleu:.2f}")


def create_model(config: Dict[str, Any]) -> SignLanguageTranslationModel:
    """
    Create the translation model from config.
    
    Args:
        config: Model configuration
    
    Returns:
        SignLanguageTranslationModel instance
    """
    encoder_type = config.get('encoder_type', 'stgcn')
    
    # Create encoder
    if encoder_type == 'stgcn':
        encoder = STGCN(
            num_joints=config.get('num_joints', 21),
            in_channels=config.get('in_channels', 3),
            hidden_channels=config.get('hidden_channels', 64),
            num_layers=config.get('encoder_layers', 6),
            output_dim=config.get('d_model', 256),
            dropout=config.get('dropout', 0.1)
        )
    elif encoder_type == 'transformer':
        encoder = MotionTransformer(
            num_joints=config.get('num_joints', 21),
            in_channels=config.get('in_channels', 3),
            d_model=config.get('d_model', 256),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('encoder_layers', 6),
            dim_feedforward=config.get('dim_feedforward', 1024),
            dropout=config.get('dropout', 0.1)
        )
    elif encoder_type == 'hybrid':
        encoder = HybridEncoder(
            num_joints=config.get('num_joints', 21),
            in_channels=config.get('in_channels', 3),
            stgcn_channels=config.get('hidden_channels', 64),
            stgcn_layers=config.get('stgcn_layers', 4),
            d_model=config.get('d_model', 256),
            num_heads=config.get('num_heads', 8),
            transformer_layers=config.get('transformer_layers', 4),
            dropout=config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    # Create full translation model
    model = SignLanguageTranslationModel(
        encoder=encoder,
        vocab_size=config.get('vocab_size', 10000),
        d_model=config.get('d_model', 256),
        num_decoder_layers=config.get('decoder_layers', 4),
        num_heads=config.get('num_heads', 8),
        dim_feedforward=config.get('dim_feedforward', 1024),
        dropout=config.get('dropout', 0.1),
        max_len=config.get('max_text_length', 512),
        pad_token_id=config.get('pad_token_id', 0),
        bos_token_id=config.get('bos_token_id', 1),
        eos_token_id=config.get('eos_token_id', 2)
    )
    
    return model


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description='Train Sign Language Translation Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    parser.add_argument('--checkpoint', type=str, help='Resume from checkpoint')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    args = parser.parse_args()
    
    # Default configuration
    config = {
        # Model
        'encoder_type': 'stgcn',  # 'stgcn', 'transformer', 'hybrid'
        'num_joints': 21,
        'in_channels': 3,
        'd_model': 256,
        'hidden_channels': 64,
        'encoder_layers': 6,
        'decoder_layers': 4,
        'num_heads': 8,
        'dim_feedforward': 1024,
        'dropout': 0.1,
        'vocab_size': 10000,
        'max_frames': 150,
        'max_text_length': 100,
        
        # Training
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'max_epochs': 100,
        'warmup_epochs': 5,
        'early_stopping_patience': 10,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'max_grad_norm': 1.0,
        'label_smoothing': 0.1,
        'use_amp': True,
        
        # Tokens
        'pad_token_id': 0,
        'bos_token_id': 1,
        'eos_token_id': 2,
    }
    
    # Load config file if provided
    if args.config and Path(args.config).exists():
        import yaml
        with open(args.config, 'r') as f:
            file_config = yaml.safe_load(f)
            config.update(file_config)
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f'run_{timestamp}'
    checkpoint_dir = run_dir / 'checkpoints'
    log_dir = run_dir / 'logs'
    
    # Save config
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create synthetic data for testing (replace with real data loading)
    logger.info("Creating synthetic training data...")
    synthetic_samples = [
        {
            'frames': [
                [
                    {'joint': f'joint_{j}', 'x': 0.1, 'y': 0.2, 'z': 0.3, 'index': j}
                    for j in range(21)
                ]
                for _ in range(60)
            ],
            'language': 'ASL',
            'gloss': 'HELLO',
            'transcript': 'Hello, how are you?'
        }
        for _ in range(100)
    ]
    
    train_samples = synthetic_samples[:80]
    val_samples = synthetic_samples[80:]
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(
        train_samples=train_samples,
        val_samples=val_samples,
        tokenizer=None,  # Use character-level for now
        batch_size=config['batch_size'],
        num_workers=0,
        max_frames=config['max_frames'],
        max_text_length=config['max_text_length'],
        num_joints=config['num_joints']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=None,
        config=config,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    # Resume from checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(Path(args.checkpoint))
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
