"""
Inference Models for SUMBA

Tracks inference requests and predictions.
"""

import uuid
from django.db import models
from django.conf import settings


class InferenceRequest(models.Model):
    """
    A single inference request.
    
    Logs all translation requests for analysis and debugging.
    """
    
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        PROCESSING = 'processing', 'Processing'
        COMPLETED = 'completed', 'Completed'
        FAILED = 'failed', 'Failed'
    
    class InputType(models.TextChoices):
        LIVE_STREAM = 'live_stream', 'Live Stream'
        UPLOADED_FILE = 'uploaded_file', 'Uploaded File'
        SAMPLE_REFERENCE = 'sample_reference', 'Sample Reference'
    
    # Identification
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    
    # Model used
    checkpoint = models.ForeignKey(
        'training.ModelCheckpoint',
        on_delete=models.SET_NULL,
        null=True,
        related_name='inference_requests'
    )
    
    # Input
    input_type = models.CharField(
        max_length=20,
        choices=InputType.choices,
        default=InputType.UPLOADED_FILE
    )
    input_file = models.FileField(upload_to='inference/inputs/', blank=True)
    input_sample = models.ForeignKey(
        'gestures.GestureSample',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='inference_requests'
    )
    
    # Input metadata
    num_frames = models.PositiveIntegerField(null=True, blank=True)
    source_language = models.ForeignKey(
        'gestures.Language',
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    
    # Output
    predicted_text = models.TextField(blank=True)
    confidence_score = models.FloatField(null=True, blank=True)
    token_confidences = models.JSONField(default=list)
    
    # Alternative predictions
    beam_results = models.JSONField(default=list, help_text="Top-k beam search results")
    
    # Status
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING
    )
    error_message = models.TextField(blank=True)
    
    # Timing
    processing_time_ms = models.FloatField(null=True, blank=True)
    
    # User tracking
    requested_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'inference_requests'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', 'created_at']),
        ]
    
    def __str__(self):
        return f"Inference {self.uuid} ({self.status})"


class InferenceFeedback(models.Model):
    """
    User feedback on inference results.
    
    Collects ground truth for model improvement.
    """
    
    class FeedbackType(models.TextChoices):
        CORRECT = 'correct', 'Correct'
        PARTIALLY_CORRECT = 'partially_correct', 'Partially Correct'
        INCORRECT = 'incorrect', 'Incorrect'
    
    inference_request = models.ForeignKey(
        InferenceRequest,
        on_delete=models.CASCADE,
        related_name='feedback'
    )
    
    feedback_type = models.CharField(max_length=20, choices=FeedbackType.choices)
    corrected_text = models.TextField(blank=True, help_text="User-provided correct translation")
    notes = models.TextField(blank=True)
    
    # User
    provided_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'inference_feedback'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Feedback on {self.inference_request.uuid}: {self.feedback_type}"


class ModelDeployment(models.Model):
    """
    Deployed model versions.
    
    Tracks which models are currently active for inference.
    """
    
    class Status(models.TextChoices):
        ACTIVE = 'active', 'Active'
        INACTIVE = 'inactive', 'Inactive'
        DEPRECATED = 'deprecated', 'Deprecated'
    
    name = models.CharField(max_length=100)
    checkpoint = models.ForeignKey(
        'training.ModelCheckpoint',
        on_delete=models.PROTECT,
        related_name='deployments'
    )
    
    # Deployment config
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.INACTIVE
    )
    is_default = models.BooleanField(default=False)
    
    # Performance metrics
    avg_inference_time_ms = models.FloatField(null=True, blank=True)
    total_requests = models.PositiveBigIntegerField(default=0)
    
    # Languages supported
    supported_languages = models.ManyToManyField(
        'gestures.Language',
        related_name='model_deployments'
    )
    
    deployed_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'model_deployments'
        ordering = ['-deployed_at']
    
    def __str__(self):
        return f"{self.name} ({self.status})"
    
    def save(self, *args, **kwargs):
        # Ensure only one default model
        if self.is_default:
            ModelDeployment.objects.filter(is_default=True).update(is_default=False)
        super().save(*args, **kwargs)
