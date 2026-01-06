"""
Core Views for SUMBA

Handles main dashboard and other core pages.
"""

from django.shortcuts import render
from django.contrib.auth.decorators import login_required

from gestures.models import GestureSample, Language
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
    
    # Global totals
    total_samples = GestureSample.objects.count()
    total_validated_samples = GestureSample.objects.filter(status=GestureSample.Status.VALIDATED).count()
    total_datasets = Dataset.objects.count()
    total_job_count = len(TRAINING_JOBS)

    return render(request, 'dashboard.html', {
        'sample_count': sample_count,
        'validated_count': validated_count,
        'dataset_count': dataset_count,
        'job_count': job_count,
        'total_samples': total_samples,
        'total_validated_samples': total_validated_samples,
        'total_datasets': total_datasets,
        'total_job_count': total_job_count,
    })


def landing(request):
    """Public landing page with live project statistics."""
    # Global counts
    total_samples = GestureSample.objects.count()
    validated_samples = GestureSample.objects.filter(status=GestureSample.Status.VALIDATED).count()
    total_datasets = Dataset.objects.count()
    job_count = len(TRAINING_JOBS)

    # User-specific counts if logged in
    your_samples = 0
    your_validated = 0
    if request.user.is_authenticated:
        your_samples = GestureSample.objects.filter(uploaded_by=request.user).count()
        your_validated = GestureSample.objects.filter(uploaded_by=request.user, status=GestureSample.Status.VALIDATED).count()

    languages = Language.objects.all()[:8]

    return render(request, 'landing.html', {
        'total_samples': total_samples,
        'validated_samples': validated_samples,
        'total_datasets': total_datasets,
        'job_count': job_count,
        'your_samples': your_samples,
        'your_validated': your_validated,
        'languages_list': languages,
    })
