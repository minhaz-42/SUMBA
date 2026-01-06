"""Admin configuration for inference app."""

from django.contrib import admin
from .models import InferenceRequest, InferenceFeedback, ModelDeployment


class InferenceFeedbackInline(admin.TabularInline):
    model = InferenceFeedback
    extra = 0
    readonly_fields = ['feedback_type', 'provided_by', 'created_at']


@admin.register(InferenceRequest)
class InferenceRequestAdmin(admin.ModelAdmin):
    list_display = [
        'uuid', 'status', 'input_type', 'source_language',
        'confidence_score', 'processing_time_ms', 'created_at'
    ]
    list_filter = ['status', 'input_type', 'source_language', 'created_at']
    search_fields = ['uuid', 'predicted_text']
    readonly_fields = ['uuid', 'created_at', 'processing_time_ms']
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Identification', {
            'fields': ('uuid', 'checkpoint')
        }),
        ('Input', {
            'fields': ('input_type', 'input_file', 'input_sample', 'num_frames', 'source_language')
        }),
        ('Output', {
            'fields': ('predicted_text', 'confidence_score', 'beam_results')
        }),
        ('Status', {
            'fields': ('status', 'error_message', 'processing_time_ms')
        }),
        ('User Info', {
            'fields': ('requested_by', 'ip_address', 'user_agent'),
            'classes': ('collapse',)
        }),
    )
    
    inlines = [InferenceFeedbackInline]


@admin.register(ModelDeployment)
class ModelDeploymentAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'checkpoint', 'status', 'is_default',
        'total_requests', 'avg_inference_time_ms', 'deployed_at'
    ]
    list_filter = ['status', 'is_default', 'deployed_at']
    search_fields = ['name']
    filter_horizontal = ['supported_languages']
