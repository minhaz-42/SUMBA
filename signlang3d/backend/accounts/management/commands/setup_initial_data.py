"""
Management command to set up initial data for the SUMBA platform.

Creates default languages, model architectures, joint definitions, and sample data.
"""

from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from gestures.models import Language, JointDefinition, GestureTag
from training.models import ModelArchitecture


User = get_user_model()


class Command(BaseCommand):
    help = 'Sets up initial data for the SUMBA platform'

    def add_arguments(self, parser):
        parser.add_argument(
            '--with-superuser',
            action='store_true',
            help='Create a superuser (uses DJANGO_SUPERUSER_PASSWORD env var)',
        )
        parser.add_argument(
            '--demo-data',
            action='store_true',
            help='Create demo datasets and samples',
        )

    def handle(self, *args, **options):
        self.stdout.write('Setting up initial data for SUMBA...\n')
        
        # Create languages
        self.create_languages()
        
        # Create joint definitions for MediaPipe hands
        self.create_joint_definitions()
        
        # Create gesture tags
        self.create_gesture_tags()
        
        # Create model architectures
        self.create_model_architectures()
        
        # Create superuser if requested
        if options['with_superuser']:
            self.create_superuser()
        
        # Create demo data if requested
        if options['demo_data']:
            self.create_demo_data()
        
        self.stdout.write(self.style.SUCCESS('\n✅ Initial data setup complete!'))

    def create_languages(self):
        """Create common sign languages."""
        languages = [
            {
                'code': 'ASL',
                'name': 'American Sign Language',
                'region': 'United States, Canada',
                'description': 'Primary sign language of Deaf communities in the United States and anglophone Canada.',
                'num_signers': 500000,
            },
            {
                'code': 'BdSL',
                'name': 'Bangladeshi Sign Language',
                'region': 'Bangladesh',
                'description': 'The sign language used by the Deaf community in Bangladesh.',
                'num_signers': 200000,
            },
            {
                'code': 'BSL',
                'name': 'British Sign Language',
                'region': 'United Kingdom',
                'description': 'The sign language used in the United Kingdom.',
                'num_signers': 150000,
            },
            {
                'code': 'ISL',
                'name': 'Indian Sign Language',
                'region': 'India',
                'description': 'The predominant sign language in South Asia.',
                'num_signers': 2700000,
            },
            {
                'code': 'JSL',
                'name': 'Japanese Sign Language',
                'region': 'Japan',
                'description': 'The sign language of the Deaf community in Japan.',
                'num_signers': 320000,
            },
            {
                'code': 'CSL',
                'name': 'Chinese Sign Language',
                'region': 'China',
                'description': 'The sign language used by Deaf communities in China.',
                'num_signers': 20000000,
            },
            {
                'code': 'DGS',
                'name': 'German Sign Language',
                'region': 'Germany',
                'description': 'The sign language of the Deaf community in Germany.',
                'num_signers': 200000,
            },
            {
                'code': 'LSF',
                'name': 'French Sign Language',
                'region': 'France',
                'description': 'The predominant sign language in France.',
                'num_signers': 100000,
            },
        ]
        
        created_count = 0
        for lang_data in languages:
            lang, created = Language.objects.update_or_create(
                code=lang_data['code'],
                defaults=lang_data
            )
            if created:
                created_count += 1
        
        self.stdout.write(f'  → Languages: {created_count} created, {len(languages) - created_count} existing')

    def create_joint_definitions(self):
        """Create MediaPipe hand landmarks."""
        # MediaPipe hand landmarks (21 per hand)
        hand_landmarks = [
            (0, 'WRIST', None),
            (1, 'THUMB_CMC', 0),
            (2, 'THUMB_MCP', 1),
            (3, 'THUMB_IP', 2),
            (4, 'THUMB_TIP', 3),
            (5, 'INDEX_FINGER_MCP', 0),
            (6, 'INDEX_FINGER_PIP', 5),
            (7, 'INDEX_FINGER_DIP', 6),
            (8, 'INDEX_FINGER_TIP', 7),
            (9, 'MIDDLE_FINGER_MCP', 0),
            (10, 'MIDDLE_FINGER_PIP', 9),
            (11, 'MIDDLE_FINGER_DIP', 10),
            (12, 'MIDDLE_FINGER_TIP', 11),
            (13, 'RING_FINGER_MCP', 0),
            (14, 'RING_FINGER_PIP', 13),
            (15, 'RING_FINGER_DIP', 14),
            (16, 'RING_FINGER_TIP', 15),
            (17, 'PINKY_MCP', 0),
            (18, 'PINKY_PIP', 17),
            (19, 'PINKY_DIP', 18),
            (20, 'PINKY_TIP', 19),
        ]
        
        # Create joints in order
        created_joints = {}
        created_count = 0
        
        for index, name, parent_index in hand_landmarks:
            parent = created_joints.get(parent_index) if parent_index is not None else None
            
            joint, created = JointDefinition.objects.update_or_create(
                index=index,
                defaults={
                    'name': name,
                    'joint_type': JointDefinition.JointType.HAND,
                    'parent_joint': parent,
                    'description': f'Hand landmark: {name.replace("_", " ").title()}',
                }
            )
            created_joints[index] = joint
            if created:
                created_count += 1
        
        self.stdout.write(f'  → Joint definitions: {created_count} created, {len(hand_landmarks) - created_count} existing')

    def create_gesture_tags(self):
        """Create common gesture tags."""
        tags = [
            ('greeting', 'Common greetings and farewells', '#10b981'),
            ('question', 'Question words and phrases', '#f59e0b'),
            ('number', 'Numbers and counting', '#6366f1'),
            ('alphabet', 'Fingerspelling alphabet', '#8b5cf6'),
            ('emotion', 'Emotional expressions', '#ef4444'),
            ('action', 'Action verbs', '#3b82f6'),
            ('family', 'Family-related signs', '#ec4899'),
            ('time', 'Time-related concepts', '#14b8a6'),
            ('food', 'Food and eating', '#f97316'),
            ('color', 'Colors', '#a855f7'),
            ('animal', 'Animals', '#22c55e'),
            ('location', 'Places and directions', '#0ea5e9'),
        ]
        
        created_count = 0
        for name, description, color in tags:
            tag, created = GestureTag.objects.update_or_create(
                name=name,
                defaults={'description': description, 'color': color}
            )
            if created:
                created_count += 1
        
        self.stdout.write(f'  → Gesture tags: {created_count} created, {len(tags) - created_count} existing')

    def create_model_architectures(self):
        """Create default model architectures."""
        architectures = [
            {
                'name': 'ST-GCN Base',
                'description': 'Spatial-Temporal Graph Convolutional Network for skeleton-based gesture recognition. Fast and efficient.',
                'encoder_type': 'stgcn',
                'decoder_type': 'transformer',
                'encoder_config': {
                    'in_channels': 3,
                    'num_classes': 1000,
                    'graph_layout': 'mediapipe_hands',
                    'graph_strategy': 'spatial',
                    'hidden_dim': 64,
                    'num_layers': 10,
                    'dropout': 0.5,
                },
                'decoder_config': {
                    'hidden_dim': 256,
                    'num_layers': 4,
                    'num_heads': 8,
                    'ff_dim': 1024,
                    'dropout': 0.1,
                    'max_length': 100,
                },
                'num_parameters': 2500000,
            },
            {
                'name': 'Motion Transformer',
                'description': 'Pure Transformer architecture for motion understanding. Better for complex gestures.',
                'encoder_type': 'transformer',
                'decoder_type': 'transformer',
                'encoder_config': {
                    'input_dim': 63,  # 21 joints * 3 coords
                    'hidden_dim': 256,
                    'num_layers': 6,
                    'num_heads': 8,
                    'ff_dim': 1024,
                    'dropout': 0.1,
                    'max_seq_len': 300,
                },
                'decoder_config': {
                    'hidden_dim': 256,
                    'num_layers': 6,
                    'num_heads': 8,
                    'ff_dim': 1024,
                    'dropout': 0.1,
                    'max_length': 100,
                },
                'num_parameters': 12000000,
            },
            {
                'name': 'Hybrid (ST-GCN + Transformer)',
                'description': 'Best of both worlds: ST-GCN spatial features + Transformer temporal modeling. Highest accuracy.',
                'encoder_type': 'hybrid',
                'decoder_type': 'transformer',
                'encoder_config': {
                    'stgcn': {
                        'in_channels': 3,
                        'hidden_dim': 64,
                        'num_layers': 6,
                    },
                    'transformer': {
                        'hidden_dim': 256,
                        'num_layers': 4,
                        'num_heads': 8,
                    },
                    'fusion': 'concat',
                },
                'decoder_config': {
                    'hidden_dim': 512,
                    'num_layers': 6,
                    'num_heads': 8,
                    'ff_dim': 2048,
                    'dropout': 0.1,
                    'max_length': 100,
                },
                'num_parameters': 25000000,
            },
        ]
        
        created_count = 0
        for arch_data in architectures:
            arch, created = ModelArchitecture.objects.update_or_create(
                name=arch_data['name'],
                defaults=arch_data
            )
            if created:
                created_count += 1
        
        self.stdout.write(f'  → Model architectures: {created_count} created, {len(architectures) - created_count} existing')

    def create_superuser(self):
        """Create a superuser for development."""
        import os
        if User.objects.filter(username='admin').exists():
            self.stdout.write('  → Superuser: already exists (admin)')
            return
        
        password = os.getenv('DJANGO_SUPERUSER_PASSWORD', 'changeme')
        user = User.objects.create_superuser(
            username=os.getenv('DJANGO_SUPERUSER_USERNAME', 'admin'),
            email=os.getenv('DJANGO_SUPERUSER_EMAIL', 'admin@sumba.local'),
            password=password,
            role='admin',
            institution='SUMBA Development',
        )
        self.stdout.write(self.style.SUCCESS('  → Superuser created (set DJANGO_SUPERUSER_PASSWORD env var)'))

    def create_demo_data(self):
        """Create demo datasets and sample data."""
        from datasets.models import Dataset, DatasetSplit
        from gestures.models import GestureSample
        
        admin = User.objects.filter(username='admin').first()
        asl = Language.objects.filter(code='ASL').first()
        
        if not admin or not asl:
            self.stdout.write(self.style.WARNING('  → Demo data: Skipped (requires superuser and ASL language)'))
            return
        
        # Create a demo dataset
        dataset, created = Dataset.objects.update_or_create(
            slug='asl-basics-demo',
            defaults={
                'name': 'ASL Basics Demo',
                'version': '1.0.0',
                'description': 'A demonstration dataset with basic ASL signs for testing the platform.',
                'created_by': admin,
                'status': 'active',
                'is_public': True,
                'license': 'CC-BY-4.0',
            }
        )
        
        if created:
            # Create splits
            DatasetSplit.objects.create(
                dataset=dataset,
                split_type='train',
                ratio=0.7,
                random_seed=42,
            )
            DatasetSplit.objects.create(
                dataset=dataset,
                split_type='validation',
                ratio=0.15,
                random_seed=42,
            )
            DatasetSplit.objects.create(
                dataset=dataset,
                split_type='test',
                ratio=0.15,
                random_seed=42,
            )
            self.stdout.write('  → Demo dataset: Created "ASL Basics Demo" with splits')
        else:
            self.stdout.write('  → Demo dataset: "ASL Basics Demo" already exists')
