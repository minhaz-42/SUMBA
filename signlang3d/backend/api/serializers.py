"""
API Serializers for SUMBA

Defines data serialization for all API endpoints.
"""

from rest_framework import serializers
from accounts.models import User, ResearcherProfile
from gestures.models import (
    Language, JointDefinition, GestureSample, 
    JointFrame, Transcript, GestureTag
)
from datasets.models import Dataset, DatasetSplit, DatasetLanguage
from training.models import (
    ModelArchitecture, TrainingRun, TrainingMetrics,
    ModelCheckpoint, EvaluationResult
)
from inference.models import InferenceRequest, InferenceFeedback, ModelDeployment


# =============================================================================
# Account Serializers
# =============================================================================

class UserSerializer(serializers.ModelSerializer):
    """Serializer for user data."""
    
    class Meta:
        model = User
        fields = [
            'id', 'username', 'email', 'role', 'institution',
            'research_interests', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class ResearcherProfileSerializer(serializers.ModelSerializer):
    """Serializer for researcher profiles."""
    
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = ResearcherProfile
        fields = [
            'user', 'orcid', 'google_scholar', 'publications_count',
            'samples_uploaded', 'samples_annotated', 'models_trained'
        ]


# =============================================================================
# Gesture Serializers
# =============================================================================

class LanguageSerializer(serializers.ModelSerializer):
    """Serializer for sign languages."""
    
    sample_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Language
        fields = [
            'id', 'code', 'name', 'description', 'region',
            'num_signers', 'sample_count', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']
    
    def get_sample_count(self, obj):
        return obj.samples.count()


class JointDefinitionSerializer(serializers.ModelSerializer):
    """Serializer for joint definitions."""
    
    class Meta:
        model = JointDefinition
        fields = ['id', 'name', 'joint_type', 'index', 'parent_joint', 'description']


class JointFrameSerializer(serializers.ModelSerializer):
    """Serializer for joint frames."""
    
    class Meta:
        model = JointFrame
        fields = ['id', 'frame_index', 'timestamp_ms', 'joints_data', 'confidence_scores']


class TranscriptSerializer(serializers.ModelSerializer):
    """Serializer for transcripts."""
    
    annotated_by_username = serializers.CharField(
        source='annotated_by.username', read_only=True
    )
    
    class Meta:
        model = Transcript
        fields = [
            'id', 'sample', 'transcript_type', 'text', 'target_language',
            'annotated_by', 'annotated_by_username', 'confidence', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class GestureSampleListSerializer(serializers.ModelSerializer):
    """Compact serializer for gesture sample lists."""
    
    language_code = serializers.CharField(source='language.code', read_only=True)
    
    class Meta:
        model = GestureSample
        fields = [
            'uuid', 'language_code', 'gloss', 'transcript', 'status',
            'num_frames', 'duration_ms', 'quality_score', 'created_at'
        ]


class GestureSampleDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer for gesture samples."""
    
    language = LanguageSerializer(read_only=True)
    language_id = serializers.PrimaryKeyRelatedField(
        queryset=Language.objects.all(),
        source='language',
        write_only=True
    )
    frames = JointFrameSerializer(many=True, read_only=True)
    transcripts = TranscriptSerializer(many=True, read_only=True)
    uploaded_by_username = serializers.CharField(
        source='uploaded_by.username', read_only=True
    )
    
    class Meta:
        model = GestureSample
        fields = [
            'uuid', 'language', 'language_id', 'gloss', 'transcript',
            'capture_method', 'num_frames', 'fps', 'duration_ms',
            'status', 'quality_score', 'frames', 'transcripts',
            'raw_data_file', 'processed_data_file',
            'uploaded_by', 'uploaded_by_username', 'validated_by',
            'recorded_at', 'created_at', 'updated_at'
        ]
        read_only_fields = ['uuid', 'created_at', 'updated_at']


class GestureSampleUploadSerializer(serializers.Serializer):
    """Serializer for gesture upload endpoint."""
    
    language = serializers.CharField()
    gloss = serializers.CharField()
    transcript = serializers.CharField(required=False, allow_blank=True)
    frames = serializers.ListField(
        child=serializers.ListField(
            child=serializers.DictField()
        )
    )
    fps = serializers.IntegerField(default=30)
    capture_method = serializers.ChoiceField(
        choices=GestureSample.CaptureMethod.choices,
        default=GestureSample.CaptureMethod.MEDIAPIPE
    )
    
    def validate_frames(self, value):
        """Validate frame structure."""
        if not value:
            raise serializers.ValidationError("At least one frame is required")
        
        for frame in value:
            for joint in frame:
                if not all(k in joint for k in ['joint', 'x', 'y', 'z']):
                    raise serializers.ValidationError(
                        "Each joint must have 'joint', 'x', 'y', 'z' fields"
                    )
        return value


# =============================================================================
# Dataset Serializers
# =============================================================================

class DatasetSplitSerializer(serializers.ModelSerializer):
    """Serializer for dataset splits."""
    
    class Meta:
        model = DatasetSplit
        fields = [
            'id', 'split_type', 'ratio', 'random_seed',
            'num_samples', 'created_at'
        ]
        read_only_fields = ['id', 'num_samples', 'created_at']


class DatasetLanguageSerializer(serializers.ModelSerializer):
    """Serializer for dataset language statistics."""
    
    language_code = serializers.CharField(source='language.code', read_only=True)
    language_name = serializers.CharField(source='language.name', read_only=True)
    
    class Meta:
        model = DatasetLanguage
        fields = [
            'id', 'language', 'language_code', 'language_name',
            'num_samples', 'num_unique_signs'
        ]


class DatasetListSerializer(serializers.ModelSerializer):
    """Compact serializer for dataset lists."""
    
    class Meta:
        model = Dataset
        fields = [
            'uuid', 'name', 'slug', 'version', 'status', 'is_public',
            'total_samples', 'vocabulary_size', 'created_at'
        ]


class DatasetDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer for datasets."""
    
    splits = DatasetSplitSerializer(many=True, read_only=True)
    languages = DatasetLanguageSerializer(many=True, read_only=True)
    created_by_username = serializers.CharField(
        source='created_by.username', read_only=True
    )
    
    class Meta:
        model = Dataset
        fields = [
            'uuid', 'name', 'slug', 'version', 'description',
            'paper_reference', 'license', 'status', 'is_public',
            'total_samples', 'total_duration_ms', 'vocabulary_size',
            'splits', 'languages', 'created_by', 'created_by_username',
            'created_at', 'updated_at', 'published_at'
        ]
        read_only_fields = [
            'uuid', 'total_samples', 'total_duration_ms', 'vocabulary_size',
            'created_at', 'updated_at'
        ]


# =============================================================================
# Training Serializers
# =============================================================================

class ModelArchitectureSerializer(serializers.ModelSerializer):
    """Serializer for model architectures."""
    
    class Meta:
        model = ModelArchitecture
        fields = [
            'id', 'name', 'description', 'encoder_type', 'decoder_type',
            'encoder_config', 'decoder_config', 'num_parameters', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class TrainingMetricsSerializer(serializers.ModelSerializer):
    """Serializer for training metrics."""
    
    class Meta:
        model = TrainingMetrics
        fields = [
            'id', 'epoch', 'step', 'train_loss', 'val_loss',
            'bleu_score', 'wer', 'cer', 'additional_metrics',
            'epoch_duration_seconds', 'created_at'
        ]


class ModelCheckpointSerializer(serializers.ModelSerializer):
    """Serializer for model checkpoints."""
    
    class Meta:
        model = ModelCheckpoint
        fields = [
            'id', 'checkpoint_type', 'epoch', 'file_path', 'file_size_bytes',
            'val_loss', 'bleu_score', 'created_at'
        ]


class TrainingRunListSerializer(serializers.ModelSerializer):
    """Compact serializer for training run lists."""
    
    architecture_name = serializers.CharField(source='architecture.name', read_only=True)
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    
    class Meta:
        model = TrainingRun
        fields = [
            'uuid', 'name', 'architecture_name', 'dataset_name',
            'status', 'current_epoch', 'max_epochs', 'created_at'
        ]


class TrainingRunDetailSerializer(serializers.ModelSerializer):
    """Detailed serializer for training runs."""
    
    architecture = ModelArchitectureSerializer(read_only=True)
    architecture_id = serializers.PrimaryKeyRelatedField(
        queryset=ModelArchitecture.objects.all(),
        source='architecture',
        write_only=True
    )
    dataset = DatasetListSerializer(read_only=True)
    dataset_id = serializers.PrimaryKeyRelatedField(
        queryset=Dataset.objects.all(),
        source='dataset',
        write_only=True
    )
    metrics = TrainingMetricsSerializer(many=True, read_only=True)
    checkpoints = ModelCheckpointSerializer(many=True, read_only=True)
    
    class Meta:
        model = TrainingRun
        fields = [
            'uuid', 'name', 'architecture', 'architecture_id',
            'dataset', 'dataset_id', 'batch_size', 'learning_rate',
            'weight_decay', 'max_epochs', 'early_stopping_patience',
            'optimizer', 'scheduler', 'random_seed', 'full_config',
            'status', 'current_epoch', 'gpu_info', 'error_message',
            'metrics', 'checkpoints', 'created_by',
            'started_at', 'completed_at', 'created_at'
        ]
        read_only_fields = [
            'uuid', 'status', 'current_epoch', 'started_at', 
            'completed_at', 'created_at'
        ]


class TrainingRunStartSerializer(serializers.Serializer):
    """Serializer for starting a training run."""
    
    name = serializers.CharField()
    architecture_id = serializers.IntegerField()
    dataset_id = serializers.IntegerField()
    batch_size = serializers.IntegerField(default=32)
    learning_rate = serializers.FloatField(default=1e-4)
    max_epochs = serializers.IntegerField(default=100)
    random_seed = serializers.IntegerField(default=42)


class EvaluationResultSerializer(serializers.ModelSerializer):
    """Serializer for evaluation results."""
    
    checkpoint_info = serializers.SerializerMethodField()
    dataset_name = serializers.CharField(source='dataset.name', read_only=True)
    language_code = serializers.CharField(source='language.code', read_only=True)
    
    class Meta:
        model = EvaluationResult
        fields = [
            'id', 'checkpoint', 'checkpoint_info', 'dataset', 'dataset_name',
            'split', 'language', 'language_code',
            'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4',
            'wer', 'cer', 'accuracy', 'f1_score',
            'num_samples_evaluated', 'inference_time_ms',
            'created_by', 'created_at'
        ]
    
    def get_checkpoint_info(self, obj):
        if obj.checkpoint:
            return {
                'training_run': obj.checkpoint.training_run.name,
                'epoch': obj.checkpoint.epoch,
                'type': obj.checkpoint.checkpoint_type
            }
        return None


# =============================================================================
# Inference Serializers
# =============================================================================

class InferenceRequestSerializer(serializers.ModelSerializer):
    """Serializer for inference requests."""
    
    class Meta:
        model = InferenceRequest
        fields = [
            'uuid', 'checkpoint', 'input_type', 'input_file', 'input_sample',
            'num_frames', 'source_language', 'predicted_text', 'confidence_score',
            'token_confidences', 'beam_results', 'status', 'error_message',
            'processing_time_ms', 'created_at'
        ]
        read_only_fields = [
            'uuid', 'predicted_text', 'confidence_score', 'token_confidences',
            'beam_results', 'status', 'error_message', 'processing_time_ms',
            'created_at'
        ]


class InferenceInputSerializer(serializers.Serializer):
    """Serializer for inference input."""
    
    frames = serializers.ListField(
        child=serializers.ListField(
            child=serializers.DictField()
        ),
        required=False
    )
    file = serializers.FileField(required=False)
    sample_uuid = serializers.UUIDField(required=False)
    language = serializers.CharField(required=False, default='ASL')
    
    def validate(self, data):
        if not any([data.get('frames'), data.get('file'), data.get('sample_uuid')]):
            raise serializers.ValidationError(
                "Must provide either 'frames', 'file', or 'sample_uuid'"
            )
        return data


class InferenceFeedbackSerializer(serializers.ModelSerializer):
    """Serializer for inference feedback."""
    
    class Meta:
        model = InferenceFeedback
        fields = [
            'id', 'inference_request', 'feedback_type', 'corrected_text',
            'notes', 'provided_by', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class ModelDeploymentSerializer(serializers.ModelSerializer):
    """Serializer for model deployments."""
    
    checkpoint_info = serializers.SerializerMethodField()
    supported_language_codes = serializers.SerializerMethodField()
    
    class Meta:
        model = ModelDeployment
        fields = [
            'id', 'name', 'checkpoint', 'checkpoint_info',
            'status', 'is_default', 'avg_inference_time_ms',
            'total_requests', 'supported_languages', 'supported_language_codes',
            'deployed_at', 'updated_at'
        ]
    
    def get_checkpoint_info(self, obj):
        return {
            'training_run': obj.checkpoint.training_run.name,
            'epoch': obj.checkpoint.epoch
        }
    
    def get_supported_language_codes(self, obj):
        return list(obj.supported_languages.values_list('code', flat=True))
