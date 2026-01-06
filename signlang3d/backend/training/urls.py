"""
URL routes for training management.
"""

from django.urls import path
from . import views

urlpatterns = [
    path('', views.training_dashboard, name='training'),
    path('start/', views.start_training, name='start_training'),
    path('config/', views.training_config, name='training_config'),
    path('job/<str:job_id>/', views.training_job, name='training_job'),
    path('job/<str:job_id>/status/', views.training_job_status, name='training_job_status'),
    path('job/<str:job_id>/cancel/', views.cancel_training, name='cancel_training'),
]
