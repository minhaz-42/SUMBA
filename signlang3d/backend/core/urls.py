"""
Core URL Configuration for SUMBA

Routes all API endpoints and serves frontend templates.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView

from . import views


urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),
    
    # API Routes
    path('api/', include('api.urls')),
    
    # Authentication
    path('accounts/', include('accounts.urls')),
    
    # Gesture Management
    path('gestures/', include('gestures.urls')),
    
    # Dataset Management
    path('datasets/', include('datasets.urls')),
    
    # Training Management
    path('training/', include('training.urls')),
    
    # Frontend Pages
    path('', TemplateView.as_view(template_name='landing.html'), name='landing'),
    path('capture/', TemplateView.as_view(template_name='capture.html'), name='capture'),
    path('demo/', TemplateView.as_view(template_name='demo.html'), name='demo'),
    path('results/', TemplateView.as_view(template_name='results.html'), name='results'),
    path('dashboard/', views.dashboard, name='dashboard'),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
