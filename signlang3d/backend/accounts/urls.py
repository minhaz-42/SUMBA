"""
URL routes for authentication.
"""

from django.urls import path
from . import views

urlpatterns = [
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile_view, name='profile'),
    path('api-keys/', views.api_keys_view, name='api_keys'),
    path('api-keys/generate/', views.generate_api_key, name='generate_api_key'),
    path('api-keys/revoke/', views.revoke_api_key, name='revoke_api_key'),
]
