"""
Core Views for SUMBA

Handles main dashboard and other core pages.
"""

from django.shortcuts import render
from django.contrib.auth.decorators import login_required

from gestures.models import GestureSample
from datasets.models import Dataset
from training.views import TRAINING_JOBS


@login_required
def dashboard(request):
    """Main dashboard with user stats and workflow."""
    user = request.user
    
    # Get user's sample stats
    sample_count = GestureSample.objects.filter(uploaded_by=user).count()
    validated_count = GestureSample.objects.filter(
        uploaded_by=user, 
        status=GestureSample.Status.VALIDATED
    ).count()
    
    # Dataset count
    dataset_count = Dataset.objects.filter(created_by=user).count()
    
    # Training job count
    job_count = len(TRAINING_JOBS)
    
    return render(request, 'dashboard.html', {
        'sample_count': sample_count,
        'validated_count': validated_count,
        'dataset_count': dataset_count,
        'job_count': job_count,
    })
