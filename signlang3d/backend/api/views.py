"""
API Views for SUMBA

REST API endpoints for gesture management, training, and inference.
"""

import logging
from datetime import datetime
from django.db.models import Count, Avg, Sum
from django.conf import settings
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter, OrderingFilter

from gestures.models import (
    Language, JointDefinition, GestureSample, 
    JointFrame, Transcript
)
from datasets.models import Dataset, DatasetSplit
from training.models import (
    ModelArchitecture, TrainingRun, TrainingMetrics,
    ModelCheckpoint, EvaluationResult
)
from inference.models import InferenceRequest, InferenceFeedback, ModelDeployment

from .serializers import (
    LanguageSerializer,
    GestureSampleListSerializer,
    GestureSampleDetailSerializer,
    GestureSampleUploadSerializer,
    TranscriptSerializer,
    DatasetListSerializer,
    DatasetDetailSerializer,
    DatasetSplitSerializer,
    ModelArchitectureSerializer,
    TrainingRunListSerializer,
    TrainingRunDetailSerializer,
    TrainingRunStartSerializer,
    EvaluationResultSerializer,
    InferenceRequestSerializer,
    InferenceInputSerializer,
    InferenceFeedbackSerializer,
    ModelDeploymentSerializer,
)

logger = logging.getLogger('ml')


# =============================================================================
# Gesture Views
# =============================================================================

class LanguageViewSet(viewsets.ModelViewSet):
    """
    API endpoint for sign languages.
    
    Supports CRUD operations and listing samples per language.
    """
    
    queryset = Language.objects.all()
    serializer_class = LanguageSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['code', 'name', 'region']
    ordering_fields = ['code', 'name', 'num_signers', 'created_at']
    
    @action(detail=True, methods=['get'])
    def samples(self, request, pk=None):
        """List all samples for a language."""
        language = self.get_object()
        samples = language.samples.filter(status=GestureSample.Status.VALIDATED)
        serializer = GestureSampleListSerializer(samples, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def statistics(self, request, pk=None):
        """Get statistics for a language."""
        language = self.get_object()
        samples = language.samples.all()
        
        stats = {
            'total_samples': samples.count(),
            'validated_samples': samples.filter(status=GestureSample.Status.VALIDATED).count(),
            'total_duration_seconds': samples.aggregate(
                total=Sum('duration_ms')
            )['total'] or 0 / 1000,
            'unique_glosses': samples.values('gloss').distinct().count(),
            'avg_frames_per_sample': samples.aggregate(
                avg=Avg('num_frames')
            )['avg'] or 0,
        }
        
        return Response(stats)


class GestureSampleViewSet(viewsets.ModelViewSet):
    """
    API endpoint for gesture samples.
    
    Supports uploading, listing, and managing 3D gesture data.
    """
    
    queryset = GestureSample.objects.select_related('language', 'uploaded_by')
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['language', 'status', 'capture_method']
    search_fields = ['gloss', 'transcript', 'uuid']
    ordering_fields = ['created_at', 'num_frames', 'duration_ms', 'quality_score']
    lookup_field = 'uuid'
    
    def get_serializer_class(self):
        if self.action == 'list':
            return GestureSampleListSerializer
        elif self.action == 'upload':
            return GestureSampleUploadSerializer
        return GestureSampleDetailSerializer
    
    @action(detail=False, methods=['post'])
    def upload(self, request):
        """
        Upload a new gesture sample with frame data.
        
        Expected format:
        {
            "language": "ASL",
            "gloss": "hello",
            "transcript": "Hello",
            "fps": 30,
            "frames": [
                [{"joint": "wrist", "x": 0.1, "y": 0.2, "z": 0.3}, ...],
                ...
            ]
        }
        """
        serializer = GestureSampleUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        data = serializer.validated_data
        
        # Get or create language
        language, _ = Language.objects.get_or_create(
            code=data['language'],
            defaults={'name': data['language']}
        )
        
        # Calculate duration
        num_frames = len(data['frames'])
        fps = data['fps']
        duration_ms = int((num_frames / fps) * 1000)
        
        # Create sample
        sample = GestureSample.objects.create(
            language=language,
            gloss=data['gloss'],
            transcript=data.get('transcript', ''),
            num_frames=num_frames,
            fps=fps,
            duration_ms=duration_ms,
            capture_method=data['capture_method'],
            uploaded_by=request.user if request.user.is_authenticated else None,
        )
        
        # Create joint frames
        joint_frames = []
        for idx, frame_joints in enumerate(data['frames']):
            joint_frames.append(JointFrame(
                sample=sample,
                frame_index=idx,
                timestamp_ms=int((idx / fps) * 1000),
                joints_data=frame_joints,
            ))
        JointFrame.objects.bulk_create(joint_frames)
        
        logger.info(f"Uploaded gesture sample: {sample.uuid}")
        
        return Response(
            GestureSampleDetailSerializer(sample).data,
            status=status.HTTP_201_CREATED
        )
    
    @action(detail=True, methods=['get'])
    def frames(self, request, uuid=None):
        """Get all frames for a gesture sample."""
        sample = self.get_object()
        frames = sample.frames.all().order_by('frame_index')
        
        return Response({
            'sample_uuid': str(sample.uuid),
            'num_frames': sample.num_frames,
            'fps': sample.fps,
            'frames': [
                {
                    'index': f.frame_index,
                    'timestamp_ms': f.timestamp_ms,
                    'joints': f.joints_data,
                    'confidence': f.confidence_scores
                }
                for f in frames
            ]
        })
    
    @action(detail=True, methods=['post'])
    def validate(self, request, uuid=None):
        """Mark a sample as validated."""
        sample = self.get_object()
        sample.status = GestureSample.Status.VALIDATED
        sample.validated_by = request.user
        sample.save()
        
        return Response({'status': 'validated'})
    
    @action(detail=True, methods=['post'])
    def reject(self, request, uuid=None):
        """Mark a sample as rejected."""
        sample = self.get_object()
        sample.status = GestureSample.Status.REJECTED
        sample.save()
        
        return Response({'status': 'rejected'})


class TranscriptViewSet(viewsets.ModelViewSet):
    """API endpoint for transcripts."""
    
    queryset = Transcript.objects.select_related('sample', 'annotated_by')
    serializer_class = TranscriptSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, SearchFilter]
    filterset_fields = ['sample', 'transcript_type', 'target_language']
    search_fields = ['text']


# =============================================================================
# Dataset Views
# =============================================================================

class DatasetViewSet(viewsets.ModelViewSet):
    """
    API endpoint for datasets.
    
    Supports dataset management and versioning.
    """
    
    queryset = Dataset.objects.prefetch_related('splits', 'languages')
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['status', 'is_public']
    search_fields = ['name', 'slug', 'description']
    ordering_fields = ['created_at', 'total_samples', 'vocabulary_size']
    lookup_field = 'slug'
    
    def get_serializer_class(self):
        if self.action == 'list':
            return DatasetListSerializer
        return DatasetDetailSerializer
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)
    
    @action(detail=True, methods=['post'])
    def update_statistics(self, request, slug=None):
        """Recalculate dataset statistics."""
        dataset = self.get_object()
        dataset.update_statistics()
        
        return Response({
            'total_samples': dataset.total_samples,
            'total_duration_ms': dataset.total_duration_ms,
            'vocabulary_size': dataset.vocabulary_size
        })
    
    @action(detail=True, methods=['post'])
    def create_splits(self, request, slug=None):
        """Create train/val/test splits for a dataset."""
        dataset = self.get_object()
        
        train_ratio = request.data.get('train_ratio', 0.8)
        val_ratio = request.data.get('val_ratio', 0.1)
        test_ratio = request.data.get('test_ratio', 0.1)
        random_seed = request.data.get('random_seed', 42)
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            return Response(
                {'error': 'Split ratios must sum to 1.0'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get sample UUIDs
        import random
        samples = list(dataset.samples.values_list('uuid', flat=True))
        random.seed(random_seed)
        random.shuffle(samples)
        
        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        splits_data = [
            (DatasetSplit.SplitType.TRAIN, train_ratio, samples[:train_end]),
            (DatasetSplit.SplitType.VALIDATION, val_ratio, samples[train_end:val_end]),
            (DatasetSplit.SplitType.TEST, test_ratio, samples[val_end:]),
        ]
        
        # Create splits
        created_splits = []
        for split_type, ratio, sample_uuids in splits_data:
            split, _ = DatasetSplit.objects.update_or_create(
                dataset=dataset,
                split_type=split_type,
                defaults={
                    'ratio': ratio,
                    'random_seed': random_seed,
                    'sample_uuids': [str(u) for u in sample_uuids],
                    'num_samples': len(sample_uuids),
                }
            )
            created_splits.append(split)
        
        return Response(DatasetSplitSerializer(created_splits, many=True).data)


class DatasetSplitViewSet(viewsets.ModelViewSet):
    """API endpoint for dataset splits."""
    
    queryset = DatasetSplit.objects.select_related('dataset')
    serializer_class = DatasetSplitSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['dataset', 'split_type']


# =============================================================================
# Training Views
# =============================================================================

class ModelArchitectureViewSet(viewsets.ModelViewSet):
    """API endpoint for model architectures."""
    
    queryset = ModelArchitecture.objects.all()
    serializer_class = ModelArchitectureSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, SearchFilter]
    filterset_fields = ['encoder_type', 'decoder_type']
    search_fields = ['name', 'description']


class TrainingRunViewSet(viewsets.ModelViewSet):
    """
    API endpoint for training runs.
    
    Supports starting, monitoring, and managing model training.
    """
    
    queryset = TrainingRun.objects.select_related(
        'architecture', 'dataset', 'created_by'
    ).prefetch_related('metrics', 'checkpoints')
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    filterset_fields = ['status', 'architecture', 'dataset']
    search_fields = ['name', 'uuid']
    ordering_fields = ['created_at', 'current_epoch']
    lookup_field = 'uuid'
    
    def get_serializer_class(self):
        if self.action == 'list':
            return TrainingRunListSerializer
        elif self.action == 'start':
            return TrainingRunStartSerializer
        return TrainingRunDetailSerializer
    
    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)
    
    @action(detail=False, methods=['post'])
    def start(self, request):
        """
        Start a new training run.
        
        This creates the training run record and would trigger
        the actual training process (via Celery or similar).
        """
        serializer = TrainingRunStartSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        data = serializer.validated_data
        
        # Create training run
        training_run = TrainingRun.objects.create(
            name=data['name'],
            architecture_id=data['architecture_id'],
            dataset_id=data['dataset_id'],
            batch_size=data['batch_size'],
            learning_rate=data['learning_rate'],
            max_epochs=data['max_epochs'],
            random_seed=data['random_seed'],
            created_by=request.user,
            status=TrainingRun.Status.PENDING,
        )
        
        # TODO: Trigger async training task
        # train_model.delay(training_run.uuid)
        
        logger.info(f"Created training run: {training_run.uuid}")
        
        return Response(
            TrainingRunDetailSerializer(training_run).data,
            status=status.HTTP_201_CREATED
        )
    
    @action(detail=True, methods=['get'])
    def metrics(self, request, uuid=None):
        """Get all metrics for a training run."""
        training_run = self.get_object()
        metrics = training_run.metrics.all().order_by('epoch')
        
        return Response({
            'training_run': str(training_run.uuid),
            'status': training_run.status,
            'current_epoch': training_run.current_epoch,
            'metrics': [
                {
                    'epoch': m.epoch,
                    'train_loss': m.train_loss,
                    'val_loss': m.val_loss,
                    'bleu_score': m.bleu_score,
                    'wer': m.wer,
                }
                for m in metrics
            ]
        })
    
    @action(detail=True, methods=['post'])
    def cancel(self, request, uuid=None):
        """Cancel a running training run."""
        training_run = self.get_object()
        
        if training_run.status not in [TrainingRun.Status.PENDING, TrainingRun.Status.RUNNING]:
            return Response(
                {'error': 'Can only cancel pending or running training runs'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        training_run.status = TrainingRun.Status.CANCELLED
        training_run.save()
        
        # TODO: Cancel async training task
        
        return Response({'status': 'cancelled'})


class EvaluationResultViewSet(viewsets.ModelViewSet):
    """API endpoint for evaluation results."""
    
    queryset = EvaluationResult.objects.select_related(
        'checkpoint__training_run', 'dataset', 'language'
    )
    serializer_class = EvaluationResultSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, OrderingFilter]
    filterset_fields = ['checkpoint', 'dataset', 'language', 'split']
    ordering_fields = ['created_at', 'bleu_4', 'wer']


# =============================================================================
# Inference Views
# =============================================================================

class InferenceViewSet(viewsets.ModelViewSet):
    """
    API endpoint for inference.
    
    Supports running model inference on gesture data.
    """
    
    queryset = InferenceRequest.objects.select_related(
        'checkpoint', 'source_language'
    )
    serializer_class = InferenceRequestSerializer
    permission_classes = [permissions.AllowAny]
    filter_backends = [DjangoFilterBackend, OrderingFilter]
    filterset_fields = ['status', 'source_language']
    ordering_fields = ['created_at', 'processing_time_ms']
    lookup_field = 'uuid'
    
    @action(detail=False, methods=['post'])
    def predict(self, request):
        """
        Run inference on gesture data.
        
        Accepts:
        - frames: Array of frame data
        - file: Uploaded file with gesture data
        - sample_uuid: Reference to existing sample
        """
        serializer = InferenceInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        data = serializer.validated_data
        
        # Get default deployment
        deployment = ModelDeployment.objects.filter(
            is_default=True,
            status=ModelDeployment.Status.ACTIVE
        ).first()
        
        if not deployment:
            return Response(
                {'error': 'No active model deployment available'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        # Create inference request
        inference_request = InferenceRequest.objects.create(
            checkpoint=deployment.checkpoint,
            input_type=self._get_input_type(data),
            source_language_id=data.get('language'),
            requested_by=request.user if request.user.is_authenticated else None,
            ip_address=self._get_client_ip(request),
            user_agent=request.META.get('HTTP_USER_AGENT', ''),
            status=InferenceRequest.Status.PENDING,
        )
        
        # TODO: Run actual inference
        # This would typically be async via Celery
        # For now, return placeholder
        
        import time
        start_time = time.time()
        
        # Placeholder inference result
        inference_request.predicted_text = "[Inference placeholder - model not connected]"
        inference_request.confidence_score = 0.0
        inference_request.status = InferenceRequest.Status.COMPLETED
        inference_request.processing_time_ms = (time.time() - start_time) * 1000
        inference_request.save()
        
        return Response(InferenceRequestSerializer(inference_request).data)
    
    def _get_input_type(self, data):
        if data.get('frames'):
            return InferenceRequest.InputType.LIVE_STREAM
        elif data.get('file'):
            return InferenceRequest.InputType.UPLOADED_FILE
        else:
            return InferenceRequest.InputType.SAMPLE_REFERENCE
    
    def _get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0]
        return request.META.get('REMOTE_ADDR')
    
    @action(detail=True, methods=['post'])
    def feedback(self, request, uuid=None):
        """Submit feedback on an inference result."""
        inference_request = self.get_object()
        
        serializer = InferenceFeedbackSerializer(data={
            **request.data,
            'inference_request': inference_request.id
        })
        serializer.is_valid(raise_exception=True)
        serializer.save(provided_by=request.user if request.user.is_authenticated else None)
        
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class ModelDeploymentViewSet(viewsets.ModelViewSet):
    """API endpoint for model deployments."""
    
    queryset = ModelDeployment.objects.select_related('checkpoint__training_run')
    serializer_class = ModelDeploymentSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['status', 'is_default']
    
    @action(detail=True, methods=['post'])
    def activate(self, request, pk=None):
        """Activate a deployment."""
        deployment = self.get_object()
        deployment.status = ModelDeployment.Status.ACTIVE
        deployment.save()
        return Response({'status': 'active'})
    
    @action(detail=True, methods=['post'])
    def set_default(self, request, pk=None):
        """Set as default deployment."""
        deployment = self.get_object()
        deployment.is_default = True
        deployment.save()
        return Response({'is_default': True})


# =============================================================================
# Utility Views
# =============================================================================

class SystemStatusView(APIView):
    """System health and status endpoint."""
    
    permission_classes = [permissions.AllowAny]
    
    def get(self, request):
        """Get system status."""
        return Response({
            'status': 'healthy',
            'version': '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'database': 'connected',
                'ml_models': 'ready',
                'websocket': 'available',
            }
        })


class StatisticsView(APIView):
    """Platform statistics endpoint."""
    
    permission_classes = [permissions.AllowAny]
    
    def get(self, request):
        """Get platform-wide statistics."""
        return Response({
            'gestures': {
                'total': GestureSample.objects.count(),
                'validated': GestureSample.objects.filter(
                    status=GestureSample.Status.VALIDATED
                ).count(),
            },
            'languages': {
                'total': Language.objects.count(),
                'breakdown': list(
                    Language.objects.annotate(
                        sample_count=Count('samples')
                    ).values('code', 'name', 'sample_count')
                ),
            },
            'datasets': {
                'total': Dataset.objects.count(),
                'public': Dataset.objects.filter(is_public=True).count(),
            },
            'training': {
                'total_runs': TrainingRun.objects.count(),
                'completed': TrainingRun.objects.filter(
                    status=TrainingRun.Status.COMPLETED
                ).count(),
            },
            'inference': {
                'total_requests': InferenceRequest.objects.count(),
                'avg_processing_time_ms': InferenceRequest.objects.filter(
                    status=InferenceRequest.Status.COMPLETED
                ).aggregate(avg=Avg('processing_time_ms'))['avg'],
            },
        })
