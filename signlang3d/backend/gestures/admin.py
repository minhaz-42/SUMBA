"""Admin configuration for gestures app."""

from django.contrib import admin
from .models import (
    Language, JointDefinition, GestureSample, 
    JointFrame, Transcript, GestureTag, GestureSampleTag
)


@admin.register(Language)
class LanguageAdmin(admin.ModelAdmin):
    list_display = ['code', 'name', 'region', 'num_signers', 'created_at']
    search_fields = ['code', 'name']


@admin.register(JointDefinition)
class JointDefinitionAdmin(admin.ModelAdmin):
    list_display = ['index', 'name', 'joint_type', 'parent_joint']
    list_filter = ['joint_type']
    ordering = ['index']


class JointFrameInline(admin.TabularInline):
    model = JointFrame
    extra = 0
    readonly_fields = ['frame_index', 'timestamp_ms']
    max_num = 10  # Limit displayed frames in admin


class TranscriptInline(admin.TabularInline):
    model = Transcript
    extra = 1


@admin.register(GestureSample)
class GestureSampleAdmin(admin.ModelAdmin):
    list_display = [
        'uuid', 'language', 'gloss', 'status', 
        'num_frames', 'capture_method', 'created_at'
    ]
    list_filter = ['language', 'status', 'capture_method', 'created_at']
    search_fields = ['uuid', 'gloss', 'transcript']
    readonly_fields = ['uuid', 'created_at', 'updated_at']
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Identification', {
            'fields': ('uuid', 'language', 'dataset')
        }),
        ('Content', {
            'fields': ('gloss', 'transcript')
        }),
        ('Technical', {
            'fields': ('capture_method', 'num_frames', 'fps', 'duration_ms')
        }),
        ('Status', {
            'fields': ('status', 'quality_score', 'validated_by')
        }),
        ('Files', {
            'fields': ('raw_data_file', 'processed_data_file'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('uploaded_by', 'recorded_at', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    inlines = [TranscriptInline]


@admin.register(GestureTag)
class GestureTagAdmin(admin.ModelAdmin):
    list_display = ['name', 'color', 'description']
    search_fields = ['name']
