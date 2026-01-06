"""
Gesture Views for SUMBA

Handles gesture library browsing, recording, and management.
"""

import json
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST, require_GET
from django.core.paginator import Paginator
from django.db.models import Q

from .models import GestureSample, JointFrame, Language
from datasets.models import Dataset


@login_required
def gesture_library(request):
    """Browse all gesture samples."""
    samples = GestureSample.objects.select_related('language', 'uploaded_by').order_by('-created_at')
    
    # Filters
    language = request.GET.get('language')
    search = request.GET.get('search')
    my_only = request.GET.get('my_only')
    
    if language:
        samples = samples.filter(language__code=language)
    
    if search:
        samples = samples.filter(
            Q(gloss__icontains=search) | Q(transcript__icontains=search)
        )
    
    if my_only == '1':
        samples = samples.filter(uploaded_by=request.user)
    
    # Pagination
    paginator = Paginator(samples, 24)
    page = request.GET.get('page', 1)
    samples_page = paginator.get_page(page)
    
    # Get all languages for filter
    languages = Language.objects.all()
    
    return render(request, 'gestures/library.html', {
        'samples': samples_page,
        'languages': languages,
        'current_language': language,
        'search': search or '',
        'my_only': my_only == '1',
    })


@login_required
def gesture_detail(request, uuid):
    """View details of a gesture sample."""
    sample = get_object_or_404(GestureSample.objects.select_related('language', 'uploaded_by'), uuid=uuid)
    
    # Get frames for visualization
    frames = sample.frames.order_by('frame_index')[:100]  # Limit for performance
    
    # Get all languages for the edit form
    languages = Language.objects.all()
    
    # Get all datasets for assignment
    datasets = Dataset.objects.filter(status='active')
    
    return render(request, 'gestures/detail.html', {
        'sample': sample,
        'frames': frames,
        'total_frames': sample.frames.count(),
        'languages': languages,
        'datasets': datasets,
    })


@login_required
def gesture_capture(request):
    """Capture page with user context."""
    languages = Language.objects.all()
    
    # Ensure default languages exist
    if not languages.exists():
        Language.objects.get_or_create(code='ASL', defaults={'name': 'American Sign Language'})
        Language.objects.get_or_create(code='BdSL', defaults={'name': 'Bangladeshi Sign Language'})
        languages = Language.objects.all()
    
    return render(request, 'gestures/capture.html', {
        'languages': languages,
    })


@login_required
@require_POST
def gesture_delete(request, uuid):
    """Delete a gesture sample."""
    sample = get_object_or_404(GestureSample, uuid=uuid)
    
    # Only allow owner or admin to delete
    if sample.uploaded_by != request.user and not request.user.is_staff:
        messages.error(request, "You don't have permission to delete this sample.")
        return redirect('gesture_library')
    
    sample.delete()
    messages.success(request, 'Gesture sample deleted successfully.')
    return redirect('gesture_library')


@login_required
def gesture_edit(request, uuid):
    """Edit gesture metadata."""
    sample = get_object_or_404(GestureSample, uuid=uuid)
    
    # Only allow owner or admin to edit
    if sample.uploaded_by != request.user and not request.user.is_staff:
        messages.error(request, "You don't have permission to edit this sample.")
        return redirect('gesture_library')
    
    if request.method == 'POST':
        sample.gloss = request.POST.get('gloss', sample.gloss)
        sample.transcript = request.POST.get('transcript', sample.transcript)
        
        language_code = request.POST.get('language')
        if language_code:
            sample.language = Language.objects.get(code=language_code)
        
        # Handle validation status
        is_validated = request.POST.get('is_validated') == 'on'
        if is_validated:
            sample.status = GestureSample.Status.VALIDATED
            sample.validated_by = request.user
        else:
            sample.status = GestureSample.Status.PENDING
            sample.validated_by = None
        
        sample.save()
        
        messages.success(request, 'Gesture updated successfully.')
        return redirect('gesture_detail', uuid=uuid)
    
    languages = Language.objects.all()
    return render(request, 'gestures/edit.html', {
        'sample': sample,
        'languages': languages,
    })


# API endpoint for saving gestures (called from WebSocket or direct POST)
@require_POST
def gesture_save_api(request):
    """Save a gesture sample via AJAX."""
    import json
    
    # Check authentication for AJAX requests
    if not request.user.is_authenticated:
        return JsonResponse({
            'error': 'Authentication required',
            'detail': 'Please log in to save gestures'
        }, status=401)
    
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    
    language_code = data.get('language', 'ASL')
    gloss = data.get('gloss', 'unlabeled')
    transcript = data.get('transcript', '')
    frames = data.get('frames', [])
    
    if not frames:
        return JsonResponse({'error': 'No frames provided'}, status=400)
    
    # Get or create language
    language, _ = Language.objects.get_or_create(
        code=language_code,
        defaults={'name': language_code}
    )
    
    # Calculate duration
    fps = data.get('fps', 30)
    duration_ms = int((len(frames) / fps) * 1000)
    
    # Create sample
    sample = GestureSample.objects.create(
        language=language,
        gloss=gloss,
        transcript=transcript,
        num_frames=len(frames),
        fps=fps,
        duration_ms=duration_ms,
        capture_method=GestureSample.CaptureMethod.MEDIAPIPE,
        uploaded_by=request.user,
    )
    
    # Create joint frames
    joint_frames = []
    for idx, frame in enumerate(frames):
        joint_frames.append(JointFrame(
            sample=sample,
            frame_index=idx,
            timestamp_ms=frame.get('timestamp', idx * (1000 // fps)),
            joints_data=frame.get('hands', frame),
            confidence_scores=frame.get('confidence')
        ))
    
    JointFrame.objects.bulk_create(joint_frames)
    
    # Update user stats if researcher
    if hasattr(request.user, 'researcher_profile'):
        request.user.researcher_profile.samples_uploaded += 1
        request.user.researcher_profile.save()
    
    return JsonResponse({
        'success': True,
        'uuid': str(sample.uuid),
        'num_frames': len(frames),
        'message': 'Gesture saved successfully'
    })


@login_required
def gesture_download(request, uuid):
    """Download gesture data as JSON."""
    sample = get_object_or_404(GestureSample, uuid=uuid)
    
    # Get all frames
    frames = list(sample.frames.order_by('frame_index').values(
        'frame_index', 'timestamp_ms', 'joints_data', 'confidence_scores'
    ))
    
    data = {
        'uuid': str(sample.uuid),
        'gloss': sample.gloss,
        'transcript': sample.transcript,
        'language': sample.language.code,
        'num_frames': sample.num_frames,
        'fps': sample.fps,
        'duration_ms': sample.duration_ms,
        'capture_method': sample.capture_method,
        'status': sample.status,
        'frames': frames,
    }
    
    response = HttpResponse(
        json.dumps(data, indent=2, default=str),
        content_type='application/json'
    )
    response['Content-Disposition'] = f'attachment; filename="{sample.gloss}_{sample.uuid}.json"'
    return response


@login_required
@require_POST
def add_sample_to_dataset(request, uuid):
    """Add a gesture sample to a dataset."""
    sample = get_object_or_404(GestureSample, uuid=uuid)
    
    dataset_uuid = request.POST.get('dataset')
    if not dataset_uuid:
        messages.error(request, 'Please select a dataset.')
        return redirect('gesture_detail', uuid=uuid)
    
    dataset = get_object_or_404(Dataset, uuid=dataset_uuid)
    
    # Assign sample to dataset
    sample.dataset = dataset
    sample.save()
    
    # Update dataset statistics
    dataset.update_statistics()
    
    messages.success(request, f'Sample added to dataset "{dataset.name}".')
    return redirect('gesture_detail', uuid=uuid)


@login_required
@require_POST
def gesture_validate(request, uuid):
    """Validate or reject a gesture sample."""
    sample = get_object_or_404(GestureSample, uuid=uuid)
    
    # Only allow staff or the uploader to validate
    if not request.user.is_staff and sample.uploaded_by != request.user:
        messages.error(request, "You don't have permission to validate this sample.")
        return redirect('gesture_detail', uuid=uuid)
    
    action = request.POST.get('action', 'validate')
    
    if action == 'validate':
        sample.status = 'validated'
        sample.validated_by = request.user
        sample.quality_score = 0.85  # Default quality score
        messages.success(request, 'Sample validated successfully.')
    elif action == 'reject':
        sample.status = 'rejected'
        sample.validated_by = request.user
        messages.warning(request, 'Sample marked as rejected.')
    elif action == 'pending':
        sample.status = 'pending'
        sample.validated_by = None
        messages.info(request, 'Sample set back to pending.')
    
    sample.save()
    return redirect('gesture_detail', uuid=uuid)


@login_required
@require_POST
def gesture_validate_bulk(request):
    """Validate multiple samples at once."""
    if not request.user.is_staff:
        return JsonResponse({'error': 'Permission denied'}, status=403)
    
    try:
        data = json.loads(request.body)
        uuids = data.get('uuids', [])
        action = data.get('action', 'validate')
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    
    if action == 'validate':
        status = 'validated'
    elif action == 'reject':
        status = 'rejected'
    else:
        status = 'pending'
    
    updated = GestureSample.objects.filter(uuid__in=uuids).update(
        status=status,
        validated_by=request.user if status != 'pending' else None,
        quality_score=0.85 if status == 'validated' else None
    )
    
    return JsonResponse({
        'success': True,
        'updated': updated,
        'status': status
    })

