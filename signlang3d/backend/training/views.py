"""
Training Views for SUMBA

Handles model training interface and job management.
"""

import os
import json
import threading
from datetime import datetime
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST, require_GET
from django.conf import settings

from datasets.models import Dataset


# In-memory training job storage (use Redis/Celery for production)
TRAINING_JOBS = {}


class TrainingJob:
    """Simple training job tracker."""
    
    def __init__(self, job_id, dataset_name, config):
        self.job_id = job_id
        self.dataset_name = dataset_name
        self.config = config
        self.status = 'pending'  # pending, running, completed, failed
        self.progress = 0
        self.current_epoch = 0
        self.total_epochs = config.get('epochs', 10)
        self.train_loss = None
        self.val_loss = None
        self.started_at = None
        self.completed_at = None
        self.error = None
        self.logs = []
    
    def to_dict(self):
        return {
            'job_id': self.job_id,
            'dataset_name': self.dataset_name,
            'config': self.config,
            'status': self.status,
            'progress': self.progress,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error': self.error,
            'logs': self.logs[-50:],  # Last 50 log entries
        }


def simulate_training(job):
    """Simulate training process (demo purposes)."""
    import time
    import random
    
    job.status = 'running'
    job.started_at = datetime.now()
    job.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Training started...")
    job.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Dataset: {job.dataset_name}")
    job.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Model: {job.config.get('model', 'ST-GCN')}")
    job.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Batch size: {job.config.get('batch_size', 32)}")
    job.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Learning rate: {job.config.get('learning_rate', 0.001)}")
    job.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Epochs: {job.total_epochs}")
    job.logs.append("")
    
    train_loss = 2.5
    val_loss = 2.8
    
    for epoch in range(1, job.total_epochs + 1):
        time.sleep(2)  # Simulate epoch time
        
        # Simulate decreasing loss
        train_loss = max(0.1, train_loss - random.uniform(0.1, 0.3))
        val_loss = max(0.15, val_loss - random.uniform(0.08, 0.25))
        
        job.current_epoch = epoch
        job.progress = int((epoch / job.total_epochs) * 100)
        job.train_loss = round(train_loss, 4)
        job.val_loss = round(val_loss, 4)
        
        job.logs.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch}/{job.total_epochs} - "
            f"Train Loss: {job.train_loss:.4f} - Val Loss: {job.val_loss:.4f}"
        )
    
    job.logs.append("")
    job.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed!")
    job.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Final Train Loss: {job.train_loss:.4f}")
    job.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Final Val Loss: {job.val_loss:.4f}")
    job.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Model saved to checkpoints/")
    
    job.status = 'completed'
    job.completed_at = datetime.now()


@login_required
def training_dashboard(request):
    """Training dashboard with job list."""
    datasets = Dataset.objects.all()
    jobs = list(TRAINING_JOBS.values())
    jobs.sort(key=lambda j: j.started_at or datetime.min, reverse=True)
    
    return render(request, 'training/dashboard.html', {
        'datasets': datasets,
        'jobs': [j.to_dict() for j in jobs],
    })


@login_required
@require_POST
def start_training(request):
    """Start a new training job."""
    dataset_uuid = request.POST.get('dataset')
    
    if not dataset_uuid:
        messages.error(request, 'Please select a dataset.')
        return redirect('training')
    
    dataset = get_object_or_404(Dataset, uuid=dataset_uuid)
    
    # Training configuration
    config = {
        'model': request.POST.get('model', 'stgcn'),
        'epochs': int(request.POST.get('epochs', 10)),
        'batch_size': int(request.POST.get('batch_size', 32)),
        'learning_rate': float(request.POST.get('learning_rate', 0.001)),
        'optimizer': request.POST.get('optimizer', 'adam'),
        'hidden_dim': int(request.POST.get('hidden_dim', 256)),
    }
    
    # Generate job ID
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create job
    job = TrainingJob(job_id, dataset.name, config)
    TRAINING_JOBS[job_id] = job
    
    # Start training in background thread (use Celery for production)
    thread = threading.Thread(target=simulate_training, args=(job,))
    thread.daemon = True
    thread.start()
    
    messages.success(request, f'Training job {job_id} started!')
    return redirect('training_job', job_id=job_id)


@login_required
def training_job(request, job_id):
    """View training job details."""
    job = TRAINING_JOBS.get(job_id)
    if not job:
        messages.error(request, 'Training job not found.')
        return redirect('training')
    
    return render(request, 'training/job.html', {
        'job': job.to_dict(),
    })


@login_required
@require_GET
def training_job_status(request, job_id):
    """Get training job status (for polling)."""
    job = TRAINING_JOBS.get(job_id)
    if not job:
        return JsonResponse({'error': 'Job not found'}, status=404)
    
    return JsonResponse(job.to_dict())


@login_required
@require_POST
def cancel_training(request, job_id):
    """Cancel a training job."""
    job = TRAINING_JOBS.get(job_id)
    if job and job.status == 'running':
        job.status = 'failed'
        job.error = 'Cancelled by user'
        job.completed_at = datetime.now()
        messages.success(request, 'Training job cancelled.')
    
    return redirect('training')


@login_required
def training_config(request):
    """Advanced training configuration page."""
    datasets = Dataset.objects.all()
    
    return render(request, 'training/config.html', {
        'datasets': datasets,
        'models': [
            {'id': 'stgcn', 'name': 'ST-GCN', 'description': 'Spatial-Temporal Graph Convolutional Network'},
            {'id': 'transformer', 'name': 'Motion Transformer', 'description': 'Transformer-based motion encoder'},
            {'id': 'hybrid', 'name': 'Hybrid (ST-GCN + Transformer)', 'description': 'Combined architecture'},
        ],
        'optimizers': [
            {'id': 'adam', 'name': 'Adam'},
            {'id': 'adamw', 'name': 'AdamW'},
            {'id': 'sgd', 'name': 'SGD'},
        ],
    })
