"""
Gesture Models for SUMBA

Core data models for 3D sign language gesture representation.
Designed for research-grade data management with full provenance tracking.
"""

import uuid
from django.db import models
from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator


class Language(models.Model):
    """
    Sign language definition.
    
    Supports multiple sign languages with metadata for research purposes.
    """
    
    code = models.CharField(max_length=10, unique=True, help_text="ISO-style code (e.g., ASL, BdSL)")
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    region = models.CharField(max_length=100, blank=True)
    
    # Statistics
    num_signers = models.PositiveIntegerField(default=0, help_text="Estimated number of users")
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'languages'
        ordering = ['code']
    
    def __str__(self):
        return f"{self.code} - {self.name}"


class JointDefinition(models.Model):
    """
    Definition of skeleton joints.
    
    Provides a standardized joint vocabulary across different capture systems.
    """
    
    class JointType(models.TextChoices):
        HAND = 'hand', 'Hand'
        ARM = 'arm', 'Arm'
        BODY = 'body', 'Body'
        FACE = 'face', 'Face'
    
    name = models.CharField(max_length=50, unique=True)
    joint_type = models.CharField(max_length=20, choices=JointType.choices)
    index = models.PositiveIntegerField(unique=True, help_text="Joint index in skeleton array")
    parent_joint = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='child_joints'
    )
    description = models.TextField(blank=True)
    
    class Meta:
        db_table = 'joint_definitions'
        ordering = ['index']
    
    def __str__(self):
        return f"{self.index}: {self.name}"


class GestureSample(models.Model):
    """
    A single gesture recording with 3D skeletal data.
    
    Core data unit for training and evaluation.
    """
    
    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending Review'
        VALIDATED = 'validated', 'Validated'
        REJECTED = 'rejected', 'Rejected'
        NEEDS_REVIEW = 'needs_review', 'Needs Review'
    
    class CaptureMethod(models.TextChoices):
        DEPTH_SENSOR = 'depth_sensor', 'Depth Sensor'
        MEDIAPIPE = 'mediapipe', 'MediaPipe'
        MOCAP = 'mocap', 'Motion Capture'
        SYNTHETIC = 'synthetic', 'Synthetic'
        OTHER = 'other', 'Other'
    
    # Unique identifier
    uuid = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    
    # Relationships
    language = models.ForeignKey(
        Language,
        on_delete=models.CASCADE,
        related_name='samples'
    )
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name='uploaded_samples'
    )
    validated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='validated_samples'
    )
    dataset = models.ForeignKey(
        'datasets.Dataset',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='samples'
    )
    
    # Transcript
    gloss = models.CharField(max_length=255, help_text="Sign gloss notation")
    transcript = models.TextField(help_text="Text translation of the gesture")
    
    # Technical metadata
    capture_method = models.CharField(
        max_length=20,
        choices=CaptureMethod.choices,
        default=CaptureMethod.MEDIAPIPE
    )
    num_frames = models.PositiveIntegerField()
    fps = models.PositiveIntegerField(default=30)
    duration_ms = models.PositiveIntegerField(help_text="Duration in milliseconds")
    
    # Quality metrics
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.PENDING
    )
    quality_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    # File storage
    raw_data_file = models.FileField(upload_to='gestures/raw/', blank=True)
    processed_data_file = models.FileField(upload_to='gestures/processed/', blank=True)
    
    # Timestamps
    recorded_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'gesture_samples'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['language', 'status']),
            models.Index(fields=['gloss']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return f"{self.language.code}: {self.gloss} ({self.num_frames} frames)"
    
    @property
    def duration_seconds(self) -> float:
        """Return duration in seconds."""
        return self.duration_ms / 1000.0
    
    @property
    def is_validated(self) -> bool:
        """Check if sample is validated."""
        return self.status == self.Status.VALIDATED
    
    @is_validated.setter
    def is_validated(self, value: bool):
        """Set validation status."""
        if value:
            self.status = self.Status.VALIDATED
        else:
            self.status = self.Status.PENDING


class JointFrame(models.Model):
    """
    A single frame of joint positions.
    
    Stores 3D coordinates for all joints at a specific timestep.
    This is the atomic unit of skeletal data.
    """
    
    sample = models.ForeignKey(
        GestureSample,
        on_delete=models.CASCADE,
        related_name='frames'
    )
    frame_index = models.PositiveIntegerField()
    timestamp_ms = models.PositiveIntegerField()
    
    # Joint data stored as JSON for flexibility
    # Format: [{"joint": "wrist", "x": 0.1, "y": 0.2, "z": 0.3}, ...]
    joints_data = models.JSONField()
    
    # Optional confidence scores per joint
    confidence_scores = models.JSONField(null=True, blank=True)
    
    class Meta:
        db_table = 'joint_frames'
        ordering = ['sample', 'frame_index']
        unique_together = ['sample', 'frame_index']
        indexes = [
            models.Index(fields=['sample', 'frame_index']),
        ]
    
    def __str__(self):
        return f"Frame {self.frame_index} of Sample {self.sample.uuid}"
    
    def get_joint_position(self, joint_name: str) -> dict | None:
        """Get position of a specific joint."""
        for joint in self.joints_data:
            if joint.get('joint') == joint_name:
                return {'x': joint['x'], 'y': joint['y'], 'z': joint['z']}
        return None


class Transcript(models.Model):
    """
    Alternative transcripts for a gesture.
    
    Supports multiple translations and annotations per gesture.
    """
    
    class TranscriptType(models.TextChoices):
        PRIMARY = 'primary', 'Primary'
        ALTERNATIVE = 'alternative', 'Alternative'
        GLOSS = 'gloss', 'Gloss Notation'
        PHONETIC = 'phonetic', 'Phonetic'
    
    sample = models.ForeignKey(
        GestureSample,
        on_delete=models.CASCADE,
        related_name='transcripts'
    )
    transcript_type = models.CharField(
        max_length=20,
        choices=TranscriptType.choices,
        default=TranscriptType.PRIMARY
    )
    text = models.TextField()
    target_language = models.CharField(
        max_length=10,
        default='en',
        help_text="Target spoken language code (e.g., en, bn)"
    )
    
    # Annotation metadata
    annotated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True
    )
    confidence = models.FloatField(
        default=1.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'transcripts'
        ordering = ['sample', '-confidence']
    
    def __str__(self):
        return f"{self.transcript_type}: {self.text[:50]}"


class GestureTag(models.Model):
    """
    Tags for categorizing gestures.
    
    Useful for filtering and research analysis.
    """
    
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True)
    color = models.CharField(max_length=7, default='#6366f1')  # Hex color
    
    class Meta:
        db_table = 'gesture_tags'
        ordering = ['name']
    
    def __str__(self):
        return self.name


class GestureSampleTag(models.Model):
    """Many-to-many relationship between gestures and tags."""
    
    sample = models.ForeignKey(
        GestureSample,
        on_delete=models.CASCADE,
        related_name='tags'
    )
    tag = models.ForeignKey(
        GestureTag,
        on_delete=models.CASCADE,
        related_name='samples'
    )
    
    class Meta:
        db_table = 'gesture_sample_tags'
        unique_together = ['sample', 'tag']
