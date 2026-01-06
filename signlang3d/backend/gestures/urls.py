"""
URL routes for gesture management.
"""

from django.urls import path
from . import views

urlpatterns = [
    path('library/', views.gesture_library, name='gesture_library'),
    path('capture/', views.gesture_capture, name='gesture_capture'),
    path('save/', views.gesture_save_api, name='gesture_save_api'),
    path('validate-bulk/', views.gesture_validate_bulk, name='gesture_validate_bulk'),
    path('<uuid:uuid>/', views.gesture_detail, name='gesture_detail'),
    path('<uuid:uuid>/edit/', views.gesture_edit, name='gesture_edit'),
    path('<uuid:uuid>/delete/', views.gesture_delete, name='gesture_delete'),
    path('<uuid:uuid>/download/', views.gesture_download, name='gesture_download'),
    path('<uuid:uuid>/add-to-dataset/', views.add_sample_to_dataset, name='add_sample_to_dataset'),
    path('<uuid:uuid>/validate/', views.gesture_validate, name='gesture_validate'),
    path('review/', views.review_dashboard, name='review_dashboard'),
    path('bulk-action/', views.gesture_bulk_action, name='gesture_bulk_action'),
]
