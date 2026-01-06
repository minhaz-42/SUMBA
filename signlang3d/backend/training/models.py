"""
Training Models for SUMBA

Tracks model training runs, checkpoints, and evaluation results.
"""

import uuid
from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator


class ModelArchitecture(models.Model):
    """
    Defines a model architecture configuration.
    
    Stores hyperparameters for reproducibility.
    """
    
    class EncoderType(models.TextChoices):
        STGCN = 'stgcn', 'ST-GCN'
        TRANSFORMER = 'transformer', 'Motion Transformer'
        HYBRID = 'hybrid', 'Hybrid ST-GCN + Transformer'
    
    class DecoderType(models.TextChoices):
        TRANSFORMER = 'transformer', 'Transformer Decoder'
        LSTM = 'lstm', 'LSTM Decoder'
        CTC = 'ctc', 'CTC Decoder'
    
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    
    # Architecture configuration
    encoder_type = models.CharField(
        max_length=20,
        choices=EncoderType.choices,
        default=EncoderType.STGCN
    )
    decoder_type = models.CharField(
        max_length=20,
        choices=DecoderType.choices,
        default=DecoderType.TRANSFORMER
    )
    
    # Hyperparameters stored as JSON
    encoder_config = models.JSONField(default=dict)
    decoder_config = models.JSONField(default=dict)
    
    # Model size
    num_parameters = models.PositiveBigIntegerField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'model_architectures'
        ordering = ['name']
    
    def __str__(self):
        return f"{self.name} ({self.encoder_type} + {self.decoder_type})"


class TrainingRun(models.Model):
    """
    A single model training run.
    
    Tracks all training metadata for full reproducibility.
    """
    
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        RUNNING = 'running', 'Running'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'
        CANCELLED = 'cancelled', 'Cancelled'
    
    # Identification
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    name = models.CharField(max_length=100)
    
    # Configuration
    architecture = models.ForeignKey(
        ModelArchitecture,
        on_delete=models.PROTECT,
        related_name='training_runs'
    )
    dataset = models.ForeignKey(
        'datasets.Dataset',
        on_delete=models.PROTECT,
        related_name='training_runs'
    )
    
    # Training hyperparameters
    batch_size = models.PositiveIntegerField(default=32)
    learning_rate = models.FloatField(default=1e-4)
    weight_decay = models.FloatField(default=1e-5)
    max_epochs = models.PositiveIntegerField(default=100)
    early_stopping_patience = models.PositiveIntegerField(default=10)
    
    # Additional config
    optimizer = models.CharField(max_length=50, default='adamw')
    scheduler = models.CharField(max_length=50, default='cosine')
    random_seed = models.PositiveIntegerField(default=42)
    
    # Full config as JSON for advanced settings
    full_config = models.JSONField(default=dict)
    
    # Status
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING
    )
    current_epoch = models.PositiveIntegerField(default=0)
    
    # User
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name='training_runs'
    )
    
    # Timing
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Hardware info
    gpu_info = models.CharField(max_length=255, blank=True)
    
    # Error tracking
    error_message = models.TextField(blank=True)
    
    class Meta:
        db_table = 'training_runs'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.status})"
    
    @property
    def duration_seconds(self):
        """Calculate training duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class TrainingMetrics(models.Model):
    """
    Metrics logged during training.
    
    Stores per-epoch metrics for visualization.
    """
    
    training_run = models.ForeignKey(
        TrainingRun,
        on_delete=models.CASCADE,
        related_name='metrics'
    )
    epoch = models.PositiveIntegerField()
    step = models.PositiveIntegerField(null=True, blank=True)
    
    # Loss metrics
    train_loss = models.FloatField()
    val_loss = models.FloatField(null=True, blank=True)
    
    # Translation metrics
    bleu_score = models.FloatField(null=True, blank=True)
    wer = models.FloatField(null=True, blank=True, help_text="Word Error Rate")
    cer = models.FloatField(null=True, blank=True, help_text="Character Error Rate")
    
    # Additional metrics as JSON
    additional_metrics = models.JSONField(default=dict)
    
    # Timing
    epoch_duration_seconds = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'training_metrics'
        ordering = ['training_run', 'epoch']
        unique_together = ['training_run', 'epoch']
    
    def __str__(self):
        return f"Epoch {self.epoch}: loss={self.train_loss:.4f}"


class ModelCheckpoint(models.Model):
    """
    Saved model checkpoints.
    
    Stores model weights at specific training points.
    """
    
    class CheckpointType(models.TextChoices):
        BEST = 'best', 'Best'
        LATEST = 'latest', 'Latest'
        EPOCH = 'epoch', 'Epoch Checkpoint'
    
    training_run = models.ForeignKey(
        TrainingRun,
        on_delete=models.CASCADE,
        related_name='checkpoints'
    )
    
    checkpoint_type = models.CharField(
        max_length=20,
        choices=CheckpointType.choices,
        default=CheckpointType.EPOCH
    )
    epoch = models.PositiveIntegerField()
    
    # File path
    file_path = models.FileField(upload_to='checkpoints/')
    file_size_bytes = models.PositiveBigIntegerField(null=True, blank=True)
    
    # Metrics at this checkpoint
    val_loss = models.FloatField(null=True, blank=True)
    bleu_score = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'model_checkpoints'
        ordering = ['training_run', '-epoch']
    
    def __str__(self):
        return f"{self.training_run.name} - {self.checkpoint_type} @ epoch {self.epoch}"


class EvaluationResult(models.Model):
    """
    Model evaluation results.
    
    Stores comprehensive evaluation metrics on test sets.
    """
    
    # Model reference
    checkpoint = models.ForeignKey(
        ModelCheckpoint,
        on_delete=models.CASCADE,
        related_name='evaluations'
    )
    
    # Evaluation dataset
    dataset = models.ForeignKey(
        'datasets.Dataset',
        on_delete=models.CASCADE,
        related_name='evaluations'
    )
    split = models.CharField(max_length=20, default='test')
    
    # Language-specific evaluation
    language = models.ForeignKey(
        'gestures.Language',
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )
    
    # Core metrics
    bleu_1 = models.FloatField(null=True, blank=True)
    bleu_2 = models.FloatField(null=True, blank=True)
    bleu_3 = models.FloatField(null=True, blank=True)
    bleu_4 = models.FloatField(null=True, blank=True)
    
    wer = models.FloatField(null=True, blank=True)
    cer = models.FloatField(null=True, blank=True)
    
    # Additional metrics
    accuracy = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    
    # Detailed results
    per_sample_results = models.JSONField(default=list)
    confusion_data = models.JSONField(default=dict)
    
    # Metadata
    num_samples_evaluated = models.PositiveIntegerField()
    inference_time_ms = models.FloatField(null=True, blank=True)
    
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'evaluation_results'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Eval: {self.checkpoint} on {self.dataset.name}"
    
    @property
    def bleu_score(self):
        """Return BLEU-4 as the default BLEU score."""
        return self.bleu_4
