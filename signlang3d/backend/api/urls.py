"""
API URL Configuration for SUMBA

Defines all REST API endpoints.
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import (
    # Gesture endpoints
    GestureSampleViewSet,
    LanguageViewSet,
    TranscriptViewSet,
    
    # Dataset endpoints
    DatasetViewSet,
    DatasetSplitViewSet,
    
    # Training endpoints
    ModelArchitectureViewSet,
    TrainingRunViewSet,
    EvaluationResultViewSet,
    
    # Inference endpoints
    InferenceViewSet,
    ModelDeploymentViewSet,
    
    # Utility endpoints
    SystemStatusView,
    StatisticsView,
)

# Create router for ViewSets
router = DefaultRouter()

# Gesture routes
router.register(r'gestures', GestureSampleViewSet, basename='gesture')
router.register(r'languages', LanguageViewSet, basename='language')
router.register(r'transcripts', TranscriptViewSet, basename='transcript')

# Dataset routes
router.register(r'datasets', DatasetViewSet, basename='dataset')
router.register(r'splits', DatasetSplitViewSet, basename='split')

# Training routes
router.register(r'architectures', ModelArchitectureViewSet, basename='architecture')
router.register(r'training', TrainingRunViewSet, basename='training')
router.register(r'evaluations', EvaluationResultViewSet, basename='evaluation')

# Inference routes
router.register(r'inference', InferenceViewSet, basename='inference')

from inference.views import LipInferenceView

# Add a simple path for lip inference (mock)
from django.urls import path

extra_urls = [
    path('inference/lip/', LipInferenceView.as_view(), name='lip_inference'),
]

urlpatterns = router.urls + extra_urls
router.register(r'deployments', ModelDeploymentViewSet, basename='deployment')

urlpatterns = [
    # Router URLs
    path('', include(router.urls)),

    # Lip inference (demo endpoint)
    path('inference/lip/', LipInferenceView.as_view(), name='lip_inference'),
    
    # Utility endpoints
    path('status/', SystemStatusView.as_view(), name='system-status'),
    path('statistics/', StatisticsView.as_view(), name='statistics'),
    
    # Authentication
    path('auth/', include('rest_framework.urls', namespace='rest_framework')),
]
