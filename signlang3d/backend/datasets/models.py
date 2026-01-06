"""
Dataset Models for SUMBA

Dataset versioning and management for research reproducibility.
"""

import uuid
from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator


class Dataset(models.Model):
    """
    A versioned collection of gesture samples.
    
    Supports dataset splitting, versioning, and research-grade metadata.
    """
    
    class Status(models.TextChoices):
        DRAFT = 'draft', 'Draft'
        ACTIVE = 'active', 'Active'
        ARCHIVED = 'archived', 'Archived'
        DEPRECATED = 'deprecated', 'Deprecated'
    
    # Identification
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    version = models.CharField(max_length=20, default='1.0.0')
    
    # Description
    description = models.TextField()
    paper_reference = models.TextField(blank=True, help_text="Citation or DOI")
    license = models.CharField(max_length=100, default='CC-BY-4.0')
    
    # Ownership
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name='created_datasets'
    )
    
    # Status
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.DRAFT
    )
    is_public = models.BooleanField(default=False)
    
    # Statistics (cached for performance)
    total_samples = models.PositiveIntegerField(default=0)
    total_duration_ms = models.PositiveBigIntegerField(default=0)
    vocabulary_size = models.PositiveIntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    published_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        db_table = 'datasets'
        ordering = ['-created_at']
        unique_together = ['name', 'version']
    
    def __str__(self):
        return f"{self.name} v{self.version}"
    
    @property
    def validated_samples_count(self):
        """Get count of validated samples dynamically."""
        from gestures.models import GestureSample
        return self.samples.filter(status=GestureSample.Status.VALIDATED).count()
    
    @property
    def pending_samples_count(self):
        """Get count of pending samples dynamically."""
        from gestures.models import GestureSample
        return self.samples.filter(status=GestureSample.Status.PENDING).count()
    
    @property
    def all_samples_count(self):
        """Get total sample count dynamically."""
        return self.samples.count()
    
    def update_statistics(self):
        """Recalculate dataset statistics from samples."""
        from gestures.models import GestureSample
        
        samples = self.samples.filter(status=GestureSample.Status.VALIDATED)
        
        self.total_samples = samples.count()
        self.total_duration_ms = sum(s.duration_ms for s in samples)
        self.vocabulary_size = samples.values('gloss').distinct().count()
        self.save(update_fields=['total_samples', 'total_duration_ms', 'vocabulary_size'])


class DatasetSplit(models.Model):
    """
    Train/validation/test splits for a dataset.
    
    Ensures reproducible experiments.
    """
    
    class SplitType(models.TextChoices):
        TRAIN = 'train', 'Training'
        VALIDATION = 'validation', 'Validation'
        TEST = 'test', 'Test'
    
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name='splits'
    )
    split_type = models.CharField(max_length=20, choices=SplitType.choices)
    
    # Split configuration
    ratio = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Fraction of dataset"
    )
    random_seed = models.PositiveIntegerField(default=42)
    
    # Samples in this split (stored as JSON array of UUIDs)
    sample_uuids = models.JSONField(default=list)
    
    # Statistics
    num_samples = models.PositiveIntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'dataset_splits'
        unique_together = ['dataset', 'split_type']
    
    def __str__(self):
        return f"{self.dataset.name} - {self.split_type} ({self.num_samples} samples)"


class DatasetLanguage(models.Model):
    """
    Languages included in a dataset with statistics.
    """
    
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name='languages'
    )
    language = models.ForeignKey(
        'gestures.Language',
        on_delete=models.CASCADE
    )
    
    # Statistics
    num_samples = models.PositiveIntegerField(default=0)
    num_unique_signs = models.PositiveIntegerField(default=0)
    
    class Meta:
        db_table = 'dataset_languages'
        unique_together = ['dataset', 'language']
    
    def __str__(self):
        return f"{self.dataset.name} - {self.language.code}"


class DatasetVersion(models.Model):
    """
    Version history for datasets.
    
    Tracks changes for reproducibility.
    """
    
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name='versions'
    )
    version = models.CharField(max_length=20)
    changelog = models.TextField()
    
    # Snapshot of statistics at this version
    total_samples = models.PositiveIntegerField()
    
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'dataset_versions'
        ordering = ['-created_at']
        unique_together = ['dataset', 'version']
    
    def __str__(self):
        return f"{self.dataset.name} changelog v{self.version}"
