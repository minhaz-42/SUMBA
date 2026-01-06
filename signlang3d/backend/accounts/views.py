"""
Authentication Views for SUMBA

Handles user registration, login, logout, and profile management.
"""

import secrets
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_exempt
from django.db import transaction

from .models import User, ResearcherProfile
from .forms import SignUpForm, LoginForm, ProfileUpdateForm


def signup_view(request):
    """Handle user registration."""
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            with transaction.atomic():
                user = form.save(commit=False)
                user.set_password(form.cleaned_data['password'])
                user.save()
                
                # Create researcher profile if researcher role
                if user.role == User.Role.RESEARCHER:
                    ResearcherProfile.objects.create(user=user)
                
                # Log in the user
                login(request, user)
                messages.success(request, f'Welcome to SUMBA, {user.username}!')
                return redirect('dashboard')
    else:
        form = SignUpForm()
    
    return render(request, 'accounts/signup.html', {'form': form})


def login_view(request):
    """Handle user login."""
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            
            # Check if login is email or username
            if '@' in username:
                try:
                    user_obj = User.objects.get(email=username)
                    username = user_obj.username
                except User.DoesNotExist:
                    pass
            
            user = authenticate(request, username=username, password=password)
            
            if user is not None:
                login(request, user)
                messages.success(request, f'Welcome back, {user.username}!')
                
                # Redirect to next page if specified
                next_url = request.GET.get('next', 'dashboard')
                return redirect(next_url)
            else:
                messages.error(request, 'Invalid username or password.')
    else:
        form = LoginForm()
    
    return render(request, 'accounts/login.html', {'form': form})


def logout_view(request):
    """Handle user logout."""
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('landing')


@login_required
def profile_view(request):
    """Display and update user profile."""
    user = request.user
    researcher_profile = None
    
    if user.role == User.Role.RESEARCHER:
        researcher_profile, _ = ResearcherProfile.objects.get_or_create(user=user)
    
    if request.method == 'POST':
        form = ProfileUpdateForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            
            # Update researcher profile if exists
            if researcher_profile:
                researcher_profile.orcid = request.POST.get('orcid', '')
                researcher_profile.google_scholar = request.POST.get('google_scholar', '')
                researcher_profile.save()
            
            messages.success(request, 'Profile updated successfully!')
            return redirect('profile')
    else:
        form = ProfileUpdateForm(instance=user)
    
    # Get user statistics
    from gestures.models import GestureSample
    from training.models import TrainingRun
    
    stats = {
        'samples_recorded': GestureSample.objects.filter(uploaded_by=user).count(),
        'training_runs': TrainingRun.objects.filter(created_by=user).count(),
    }
    
    return render(request, 'accounts/profile.html', {
        'form': form,
        'researcher_profile': researcher_profile,
        'stats': stats,
    })


@login_required
def api_keys_view(request):
    """Manage API keys for programmatic access."""
    user = request.user
    
    return render(request, 'accounts/api_keys.html', {
        'api_key': user.api_key,
        'api_requests_count': user.api_requests_count,
    })


@login_required
@require_POST
def generate_api_key(request):
    """Generate a new API key for the user."""
    user = request.user
    user.api_key = secrets.token_hex(32)
    user.save()
    
    messages.success(request, 'New API key generated successfully!')
    return redirect('api_keys')


@login_required
@require_POST
def revoke_api_key(request):
    """Revoke the current API key."""
    user = request.user
    user.api_key = None
    user.save()
    
    messages.warning(request, 'API key revoked.')
    return redirect('api_keys')


# API Authentication Middleware Helper
def get_user_from_api_key(api_key: str):
    """Validate API key and return user."""
    if not api_key:
        return None
    
    try:
        user = User.objects.get(api_key=api_key)
        user.api_requests_count += 1
        user.save(update_fields=['api_requests_count'])
        return user
    except User.DoesNotExist:
        return None
