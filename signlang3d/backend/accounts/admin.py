"""Admin configuration for accounts app."""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User, ResearcherProfile


class ResearcherProfileInline(admin.StackedInline):
    model = ResearcherProfile
    can_delete = False
    verbose_name_plural = 'Researcher Profile'


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ['username', 'email', 'role', 'institution', 'is_active', 'created_at']
    list_filter = ['role', 'is_active', 'is_staff', 'created_at']
    search_fields = ['username', 'email', 'institution']
    ordering = ['-created_at']
    
    fieldsets = BaseUserAdmin.fieldsets + (
        ('Research Info', {
            'fields': ('role', 'institution', 'research_interests')
        }),
        ('API Access', {
            'fields': ('api_key', 'api_requests_count')
        }),
    )
    
    inlines = [ResearcherProfileInline]


@admin.register(ResearcherProfile)
class ResearcherProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'orcid', 'samples_uploaded', 'models_trained']
    search_fields = ['user__username', 'orcid']
