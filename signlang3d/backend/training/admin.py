"""Admin configuration for training app."""

from django.contrib import admin
from .models import (
    ModelArchitecture, TrainingRun, TrainingMetrics,
    ModelCheckpoint, EvaluationResult
)


@admin.register(ModelArchitecture)
class ModelArchitectureAdmin(admin.ModelAdmin):
    list_display = ['name', 'encoder_type', 'decoder_type', 'num_parameters', 'created_at']
    list_filter = ['encoder_type', 'decoder_type']
    search_fields = ['name', 'description']


class TrainingMetricsInline(admin.TabularInline):
    model = TrainingMetrics
    extra = 0
    readonly_fields = ['epoch', 'train_loss', 'val_loss', 'bleu_score', 'wer']
    max_num = 20


class ModelCheckpointInline(admin.TabularInline):
    model = ModelCheckpoint
    extra = 0
    readonly_fields = ['checkpoint_type', 'epoch', 'val_loss', 'bleu_score', 'created_at']


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'architecture', 'dataset', 'status',
        'current_epoch', 'max_epochs', 'created_by', 'created_at'
    ]
    list_filter = ['status', 'architecture', 'created_at']
    search_fields = ['name', 'uuid']
    readonly_fields = ['uuid', 'started_at', 'completed_at', 'created_at']
    date_hierarchy = 'created_at'
    
    fieldsets = (
        ('Identification', {
            'fields': ('uuid', 'name')
        }),
        ('Configuration', {
            'fields': ('architecture', 'dataset')
        }),
        ('Hyperparameters', {
            'fields': (
                'batch_size', 'learning_rate', 'weight_decay',
                'max_epochs', 'early_stopping_patience',
                'optimizer', 'scheduler', 'random_seed'
            )
        }),
        ('Status', {
            'fields': ('status', 'current_epoch', 'error_message')
        }),
        ('Hardware', {
            'fields': ('gpu_info',),
            'classes': ('collapse',)
        }),
        ('Timing', {
            'fields': ('created_by', 'started_at', 'completed_at', 'created_at'),
            'classes': ('collapse',)
        }),
    )
    
    inlines = [TrainingMetricsInline, ModelCheckpointInline]


@admin.register(EvaluationResult)
class EvaluationResultAdmin(admin.ModelAdmin):
    list_display = [
        'checkpoint', 'dataset', 'language', 'split',
        'bleu_4', 'wer', 'num_samples_evaluated', 'created_at'
    ]
    list_filter = ['split', 'language', 'created_at']
    search_fields = ['checkpoint__training_run__name']
    readonly_fields = ['created_at']
