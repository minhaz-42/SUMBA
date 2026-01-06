"""Gestures app configuration."""

from django.apps import AppConfig


class GesturesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'gestures'
    verbose_name = 'Gesture Data'
