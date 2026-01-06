"""
URL routes for dataset management.
"""

from django.urls import path
from . import views

urlpatterns = [
    path('', views.dataset_list, name='datasets'),
    path('create/', views.dataset_create, name='dataset_create'),
    path('upload/', views.dataset_upload, name='dataset_upload'),
    path('<uuid:uuid>/', views.dataset_detail, name='dataset_detail'),
    path('<uuid:uuid>/add-samples/', views.dataset_add_samples, name='dataset_add_samples'),
    path('<uuid:uuid>/auto-split/', views.dataset_auto_split, name='dataset_auto_split'),
    path('<uuid:uuid>/export/', views.dataset_export, name='dataset_export'),
    path('<uuid:uuid>/delete/', views.dataset_delete, name='dataset_delete'),
]
