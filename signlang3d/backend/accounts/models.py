"""
User Models for SUMBA

Custom user model with research-specific fields.
"""

from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """
    Extended user model for research platform.
    
    Includes fields for researcher profiles and institutional affiliations.
    """
    
    class Role(models.TextChoices):
        RESEARCHER = 'researcher', 'Researcher'
        ANNOTATOR = 'annotator', 'Annotator'
        ADMIN = 'admin', 'Administrator'
        VIEWER = 'viewer', 'Viewer'
    
    email = models.EmailField(unique=True)
    role = models.CharField(
        max_length=20,
        choices=Role.choices,
        default=Role.VIEWER
    )
    institution = models.CharField(max_length=255, blank=True)
    research_interests = models.TextField(blank=True)
    
    # API access
    api_key = models.CharField(max_length=64, blank=True, unique=True, null=True)
    api_requests_count = models.PositiveIntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'users'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.username} ({self.role})"
    
    def can_train_models(self) -> bool:
        """Check if user has permission to train models."""
        return self.role in [self.Role.RESEARCHER, self.Role.ADMIN]
    
    def can_annotate(self) -> bool:
        """Check if user has permission to annotate data."""
        return self.role in [self.Role.RESEARCHER, self.Role.ANNOTATOR, self.Role.ADMIN]


class ResearcherProfile(models.Model):
    """
    Extended profile for researchers.
    
    Contains publication history and contribution metrics.
    """
    
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='researcher_profile'
    )
    orcid = models.CharField(max_length=50, blank=True, help_text="ORCID identifier")
    google_scholar = models.URLField(blank=True)
    publications_count = models.PositiveIntegerField(default=0)
    
    # Contribution tracking
    samples_uploaded = models.PositiveIntegerField(default=0)
    samples_annotated = models.PositiveIntegerField(default=0)
    models_trained = models.PositiveIntegerField(default=0)
    
    class Meta:
        db_table = 'researcher_profiles'
    
    def __str__(self):
        return f"Profile: {self.user.username}"
