"""Admin configuration for datasets app."""

from django.contrib import admin
from .models import Dataset, DatasetSplit, DatasetLanguage, DatasetVersion


class DatasetSplitInline(admin.TabularInline):
    model = DatasetSplit
    extra = 0
    readonly_fields = ['num_samples']


class DatasetLanguageInline(admin.TabularInline):
    model = DatasetLanguage
    extra = 0
    readonly_fields = ['num_samples', 'num_unique_signs']


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'version', 'status', 'is_public',
        'total_samples', 'vocabulary_size', 'created_at'
    ]
    list_filter = ['status', 'is_public', 'created_at']
    search_fields = ['name', 'slug', 'description']
    prepopulated_fields = {'slug': ('name',)}
    readonly_fields = [
        'uuid', 'total_samples', 'total_duration_ms',
        'vocabulary_size', 'created_at', 'updated_at'
    ]
    
    fieldsets = (
        ('Identification', {
            'fields': ('uuid', 'name', 'slug', 'version')
        }),
        ('Description', {
            'fields': ('description', 'paper_reference', 'license')
        }),
        ('Status', {
            'fields': ('status', 'is_public', 'created_by')
        }),
        ('Statistics', {
            'fields': ('total_samples', 'total_duration_ms', 'vocabulary_size'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at', 'published_at'),
            'classes': ('collapse',)
        }),
    )
    
    inlines = [DatasetSplitInline, DatasetLanguageInline]
    
    actions = ['update_statistics']
    
    @admin.action(description='Update statistics for selected datasets')
    def update_statistics(self, request, queryset):
        for dataset in queryset:
            dataset.update_statistics()
        self.message_user(request, f'Updated statistics for {queryset.count()} datasets')


@admin.register(DatasetVersion)
class DatasetVersionAdmin(admin.ModelAdmin):
    list_display = ['dataset', 'version', 'total_samples', 'created_by', 'created_at']
    list_filter = ['created_at']
    search_fields = ['dataset__name', 'changelog']
