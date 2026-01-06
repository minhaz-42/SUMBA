"""
WebSocket routing for gesture capture.
"""

from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/gesture/capture/$', consumers.GestureCaptureConsumer.as_asgi()),
    re_path(r'ws/gesture/inference/$', consumers.InferenceConsumer.as_asgi()),
]
