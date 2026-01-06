"""
Dataset Management Views for SUMBA

Handles dataset creation, browsing, and split management.
"""

import json
import random
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.core.paginator import Paginator
from django.db.models import Count
from django.utils.text import slugify

from .models import Dataset, DatasetSplit, DatasetLanguage
from gestures.models import GestureSample, Language


@login_required
def dataset_list(request):
    """List all datasets."""
    datasets = Dataset.objects.select_related('created_by').order_by('-created_at')
    
    # Filter by status
    status = request.GET.get('status')
    if status:
        datasets = datasets.filter(status=status)
    
    languages = Language.objects.all()
    
    return render(request, 'datasets/list.html', {
        'datasets': datasets,
        'languages': languages,
        'current_status': status,
    })


@login_required
def dataset_detail(request, uuid):
    """View dataset details and samples."""
    dataset = get_object_or_404(Dataset.objects.select_related('created_by'), uuid=uuid)
    
    splits = dataset.splits.all()
    
    # Group by split type
    train_split = splits.filter(split_type='train').first()
    val_split = splits.filter(split_type='validation').first()
    test_split = splits.filter(split_type='test').first()
    
    return render(request, 'datasets/detail.html', {
        'dataset': dataset,
        'train_split': train_split,
        'val_split': val_split,
        'test_split': test_split,
        'train_count': train_split.num_samples if train_split else 0,
        'val_count': val_split.num_samples if val_split else 0,
        'test_count': test_split.num_samples if test_split else 0,
    })


@login_required
def dataset_create(request):
    """Create a new dataset."""
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description', '')
        version = request.POST.get('version', '1.0.0')
        
        dataset = Dataset.objects.create(
            name=name,
            slug=slugify(name),
            description=description,
            version=version,
            created_by=request.user,
        )
        
        messages.success(request, f'Dataset "{name}" created successfully!')
        return redirect('dataset_detail', uuid=dataset.uuid)
    
    languages = Language.objects.all()
    return render(request, 'datasets/create.html', {
        'languages': languages,
    })


@login_required
def dataset_add_samples(request, uuid):
    """Add gesture samples to a dataset."""
    dataset = get_object_or_404(Dataset, uuid=uuid)
    
    if request.method == 'POST':
        sample_uuids = request.POST.getlist('samples')
        split_type = request.POST.get('split', 'train')
        
        # Get or create split
        split, created = DatasetSplit.objects.get_or_create(
            dataset=dataset,
            split_type=split_type,
            defaults={'ratio': 0.8 if split_type == 'train' else 0.1, 'random_seed': 42}
        )
        
        # Add UUIDs to split
        existing = set(split.sample_uuids or [])
        added = 0
        for sample_uuid in sample_uuids:
            if sample_uuid not in existing:
                existing.add(sample_uuid)
                added += 1
        
        split.sample_uuids = list(existing)
        split.num_samples = len(split.sample_uuids)
        split.save()
        
        # Update dataset stats
        dataset.total_samples = sum(s.num_samples for s in dataset.splits.all())
        dataset.save()
        
        messages.success(request, f'Added {added} samples to {split_type} split.')
        return redirect('dataset_detail', uuid=uuid)
    
    # Get available samples
    available_samples = GestureSample.objects.filter(
        status=GestureSample.Status.VALIDATED
    ).order_by('gloss')[:100]
    
    return render(request, 'datasets/add_samples.html', {
        'dataset': dataset,
        'samples': available_samples,
    })


@login_required
@require_POST
def dataset_auto_split(request, uuid):
    """Automatically create train/val/test splits."""
    dataset = get_object_or_404(Dataset, uuid=uuid)
    
    train_ratio = float(request.POST.get('train_ratio', 0.8))
    val_ratio = float(request.POST.get('val_ratio', 0.1))
    
    # Get all sample UUIDs from all splits
    all_uuids = []
    for split in dataset.splits.all():
        all_uuids.extend(split.sample_uuids or [])
    
    all_uuids = list(set(all_uuids))
    random.shuffle(all_uuids)
    
    n = len(all_uuids)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Create/update splits
    train_split, _ = DatasetSplit.objects.update_or_create(
        dataset=dataset, split_type='train',
        defaults={'ratio': train_ratio, 'sample_uuids': all_uuids[:train_end], 'num_samples': train_end}
    )
    val_split, _ = DatasetSplit.objects.update_or_create(
        dataset=dataset, split_type='validation',
        defaults={'ratio': val_ratio, 'sample_uuids': all_uuids[train_end:val_end], 'num_samples': val_end - train_end}
    )
    test_split, _ = DatasetSplit.objects.update_or_create(
        dataset=dataset, split_type='test',
        defaults={'ratio': 1 - train_ratio - val_ratio, 'sample_uuids': all_uuids[val_end:], 'num_samples': n - val_end}
    )
    
    messages.success(request, f'Created splits: {train_end} train, {val_end - train_end} val, {n - val_end} test')
    return redirect('dataset_detail', uuid=uuid)


@login_required
@require_POST
def dataset_delete(request, uuid):
    """Delete a dataset."""
    dataset = get_object_or_404(Dataset, uuid=uuid)
    
    if dataset.created_by != request.user and not request.user.is_staff:
        messages.error(request, "You don't have permission to delete this dataset.")
        return redirect('datasets')
    
    name = dataset.name
    dataset.delete()
    messages.success(request, f'Dataset "{name}" deleted.')
    return redirect('datasets')


@login_required
def dataset_export(request, uuid):
    """Export dataset as JSON."""
    dataset = get_object_or_404(Dataset, uuid=uuid)
    
    data = {
        'name': dataset.name,
        'version': dataset.version,
        'description': dataset.description,
        'splits': {}
    }
    
    for split in dataset.splits.all():
        data['splits'][split.split_type] = {
            'ratio': split.ratio,
            'sample_uuids': split.sample_uuids,
            'num_samples': split.num_samples,
        }
    
    return JsonResponse(data)


@login_required
def dataset_upload(request):
    """Upload samples from JSON/CSV file."""
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        dataset_uuid = request.POST.get('dataset_uuid')
        
        if not uploaded_file:
            messages.error(request, 'No file uploaded.')
            return redirect('datasets')
        
        try:
            content = uploaded_file.read().decode('utf-8')
            data = json.loads(content)
            
            # Handle different formats
            if 'samples' in data:
                samples = data['samples']
            elif 'frames' in data:
                samples = [data]  # Single sample
            else:
                samples = data if isinstance(data, list) else [data]
            
            # Create gesture samples
            language_code = data.get('language', 'ASL')
            language, _ = Language.objects.get_or_create(
                code=language_code,
                defaults={'name': language_code}
            )
            
            created = 0
            for sample in samples:
                frames = sample.get('frames', [])
                if not frames:
                    continue
                
                gesture = GestureSample.objects.create(
                    language=language,
                    gloss=sample.get('gloss', 'uploaded'),
                    transcript=sample.get('transcript', ''),
                    num_frames=len(frames),
                    fps=sample.get('fps', 30),
                    duration_ms=int(len(frames) / sample.get('fps', 30) * 1000),
                    capture_method='upload',
                    uploaded_by=request.user,
                )
                
                # If dataset specified, add to its train split
                if dataset_uuid:
                    try:
                        dataset = Dataset.objects.get(uuid=dataset_uuid)
                        split, _ = DatasetSplit.objects.get_or_create(
                            dataset=dataset, split_type='train',
                            defaults={'ratio': 0.8}
                        )
                        uuids = split.sample_uuids or []
                        uuids.append(str(gesture.uuid))
                        split.sample_uuids = uuids
                        split.num_samples = len(uuids)
                        split.save()
                    except Dataset.DoesNotExist:
                        pass
                
                created += 1
            
            messages.success(request, f'Imported {created} gesture samples.')
            
        except json.JSONDecodeError:
            messages.error(request, 'Invalid JSON file.')
        except Exception as e:
            messages.error(request, f'Import error: {str(e)}')
        
        return redirect('datasets')
    
    datasets = Dataset.objects.all()
    return render(request, 'datasets/upload.html', {
        'datasets': datasets,
    })
