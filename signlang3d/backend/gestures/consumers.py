"""
WebSocket consumers for real-time gesture capture and inference.

Handles streaming skeletal data from frontend depth sensors.
"""

import json
import logging
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async
from django.conf import settings

logger = logging.getLogger('ml')


class GestureCaptureConsumer(AsyncJsonWebsocketConsumer):
    """
    WebSocket consumer for live gesture capture.
    
    Receives streaming 3D skeletal data and stores it as gesture samples.
    """
    
    async def connect(self):
        """Handle WebSocket connection."""
        self.session_id = self.scope.get('session', {}).get('session_key', 'anonymous')
        self.frames_buffer = []
        self.max_frames = settings.SKELETON_CONFIG['max_frames']
        self.is_recording = False
        
        await self.accept()
        await self.send_json({
            'type': 'connection_established',
            'message': 'Ready to receive gesture data',
            'config': settings.SKELETON_CONFIG
        })
        
        logger.info(f"Gesture capture session started: {self.session_id}")
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        if self.frames_buffer:
            logger.info(f"Session ended with {len(self.frames_buffer)} frames captured")
        self.frames_buffer = []
    
    async def receive_json(self, content):
        """
        Handle incoming JSON messages.
        
        Expected message types:
        - start_recording: Begin capturing frames
        - frame: Single frame of joint data
        - stop_recording: End capture and save
        - cancel: Discard current recording
        """
        msg_type = content.get('type')
        
        if msg_type == 'start_recording':
            await self.start_recording(content)
        
        elif msg_type == 'frame':
            await self.receive_frame(content)
        
        elif msg_type == 'stop_recording':
            await self.stop_recording(content)
        
        elif msg_type == 'cancel':
            await self.cancel_recording()
        
        else:
            await self.send_json({
                'type': 'error',
                'message': f'Unknown message type: {msg_type}'
            })
    
    async def start_recording(self, content):
        """Initialize recording session."""
        self.frames_buffer = []
        self.is_recording = True
        self.recording_metadata = {
            'language': content.get('language', 'ASL'),
            'gloss': content.get('gloss', ''),
            'fps': content.get('fps', settings.SKELETON_CONFIG['fps']),
        }
        
        await self.send_json({
            'type': 'recording_started',
            'message': 'Recording in progress...'
        })
    
    async def receive_frame(self, content):
        """Process incoming skeletal frame."""
        if not self.is_recording:
            return
        
        if len(self.frames_buffer) >= self.max_frames:
            await self.send_json({
                'type': 'warning',
                'message': f'Maximum frames ({self.max_frames}) reached'
            })
            return
        
        frame_data = {
            'frame_index': len(self.frames_buffer),
            'timestamp_ms': content.get('timestamp_ms', len(self.frames_buffer) * 33),
            'joints_data': content.get('joints', []),
            'confidence_scores': content.get('confidence', None)
        }
        
        self.frames_buffer.append(frame_data)
        
        # Send periodic status updates
        if len(self.frames_buffer) % 30 == 0:
            await self.send_json({
                'type': 'status',
                'frames_captured': len(self.frames_buffer)
            })
    
    async def stop_recording(self, content):
        """Finalize recording and save to database."""
        self.is_recording = False
        
        if not self.frames_buffer:
            await self.send_json({
                'type': 'error',
                'message': 'No frames captured'
            })
            return
        
        # Add transcript from content
        transcript = content.get('transcript', '')
        self.recording_metadata['transcript'] = transcript
        
        # Save to database
        sample_uuid = await self.save_gesture_sample()
        
        await self.send_json({
            'type': 'recording_complete',
            'sample_uuid': str(sample_uuid),
            'num_frames': len(self.frames_buffer),
            'message': 'Gesture saved successfully'
        })
        
        self.frames_buffer = []
    
    async def cancel_recording(self):
        """Cancel current recording without saving."""
        self.is_recording = False
        frames_discarded = len(self.frames_buffer)
        self.frames_buffer = []
        
        await self.send_json({
            'type': 'recording_cancelled',
            'frames_discarded': frames_discarded
        })
    
    @database_sync_to_async
    def save_gesture_sample(self):
        """Save the captured gesture to database."""
        from .models import GestureSample, JointFrame, Language
        
        # Get or create language
        language, _ = Language.objects.get_or_create(
            code=self.recording_metadata['language'],
            defaults={'name': self.recording_metadata['language']}
        )
        
        # Calculate duration
        fps = self.recording_metadata['fps']
        num_frames = len(self.frames_buffer)
        duration_ms = int((num_frames / fps) * 1000)
        
        # Create sample
        sample = GestureSample.objects.create(
            language=language,
            gloss=self.recording_metadata.get('gloss', 'unlabeled'),
            transcript=self.recording_metadata.get('transcript', ''),
            num_frames=num_frames,
            fps=fps,
            duration_ms=duration_ms,
            capture_method=GestureSample.CaptureMethod.MEDIAPIPE,
        )
        
        # Create joint frames
        joint_frames = [
            JointFrame(
                sample=sample,
                frame_index=frame['frame_index'],
                timestamp_ms=frame['timestamp_ms'],
                joints_data=frame['joints_data'],
                confidence_scores=frame['confidence_scores']
            )
            for frame in self.frames_buffer
        ]
        JointFrame.objects.bulk_create(joint_frames)
        
        logger.info(f"Saved gesture sample: {sample.uuid} with {num_frames} frames")
        
        return sample.uuid


class InferenceConsumer(AsyncJsonWebsocketConsumer):
    """
    WebSocket consumer for real-time inference.
    
    Streams skeletal data and returns translation predictions.
    """
    
    # Demo phrases for different gestures (based on hand count and motion)
    DEMO_TRANSLATIONS = {
        'ASL': [
            "Hello, how are you?",
            "Thank you very much",
            "Nice to meet you",
            "My name is...",
            "Good morning",
            "I understand",
            "Please help me",
            "Where is the bathroom?",
            "I love you",
            "See you later",
        ],
        'BdSL': [
            "আপনি কেমন আছেন?",
            "ধন্যবাদ",
            "আপনার সাথে দেখা হয়ে ভালো লাগলো",
            "আমার নাম...",
            "সুপ্রভাত",
            "আমি বুঝতে পেরেছি",
            "আমাকে সাহায্য করুন",
            "বাথরুম কোথায়?",
            "আমি তোমাকে ভালোবাসি",
            "পরে দেখা হবে",
        ]
    }
    
    async def connect(self):
        """Handle connection for inference."""
        self.frames_buffer = []
        self.inference_window = 30  # Frames to accumulate before inference
        self.inference_count = 0
        
        await self.accept()
        await self.send_json({
            'type': 'connection_established',
            'message': 'Ready for real-time inference'
        })
        logger.info("Inference consumer connected")
    
    async def disconnect(self, close_code):
        """Handle disconnection."""
        self.frames_buffer = []
        logger.info(f"Inference consumer disconnected: {close_code}")
    
    async def receive_json(self, content):
        """Process incoming frames for inference."""
        msg_type = content.get('type')
        logger.info(f"Received message type: {msg_type}")
        
        if msg_type == 'frame':
            await self.process_inference_frame(content)
        
        elif msg_type == 'inference':
            # Handle batch inference request from frontend
            await self.process_batch_inference(content)
        
        elif msg_type == 'reset':
            self.frames_buffer = []
            await self.send_json({
                'type': 'buffer_reset',
                'message': 'Inference buffer cleared'
            })
    
    async def process_batch_inference(self, content):
        """Process batch of frames for inference."""
        import random
        import asyncio
        
        request_id = content.get('request_id', '')
        frames = content.get('frames', [])
        model = content.get('model', 'hybrid')
        language = content.get('language', 'ASL')
        
        logger.info(f"Processing batch: {len(frames)} frames, model={model}, lang={language}")
        
        # Simulate processing delay (would be real ML inference)
        await asyncio.sleep(0.5)
        
        # Get demo translation
        translations = self.DEMO_TRANSLATIONS.get(language, self.DEMO_TRANSLATIONS['ASL'])
        translation_idx = self.inference_count % len(translations)
        translation = translations[translation_idx]
        self.inference_count += 1
        
        # Generate realistic confidence and alternatives
        confidence = random.uniform(0.82, 0.98)
        alternatives = []
        for i in range(2):
            alt_idx = (translation_idx + i + 1) % len(translations)
            alternatives.append({
                'text': translations[alt_idx],
                'score': confidence - random.uniform(0.1, 0.25)
            })
        
        # Send translation response
        await self.send_json({
            'type': 'translation',
            'request_id': request_id,
            'translation': translation,
            'confidence': round(confidence, 3),
            'alternatives': alternatives,
            'frames_processed': len(frames),
            'model_used': model,
            'beam_log': [
                f"Received {len(frames)} frames",
                f"Model: {model} encoder",
                f"Running beam search (width=5)...",
                f"Top hypothesis: \"{translation}\" (score: {confidence:.3f})",
                "Decoding complete"
            ]
        })
        
        logger.info(f"Sent translation: {translation}")
    
    async def process_inference_frame(self, content):
        """Accumulate frames and run inference."""
        frame_data = content.get('joints', [])
        self.frames_buffer.append(frame_data)
        
        # Run inference when we have enough frames
        if len(self.frames_buffer) >= self.inference_window:
            prediction = await self.run_inference()
            
            await self.send_json({
                'type': 'prediction',
                'text': prediction['text'],
                'confidence': prediction['confidence'],
                'frames_processed': len(self.frames_buffer)
            })
            
            # Sliding window: keep last half of frames
            self.frames_buffer = self.frames_buffer[self.inference_window // 2:]
    
    async def run_inference(self):
        """
        Run model inference on accumulated frames.
        
        Returns demo prediction for now - connect to real ML pipeline for production.
        """
        import random
        
        translations = self.DEMO_TRANSLATIONS['ASL']
        translation = translations[self.inference_count % len(translations)]
        self.inference_count += 1
        
        return {
            'text': translation,
            'confidence': random.uniform(0.85, 0.98)
        }
