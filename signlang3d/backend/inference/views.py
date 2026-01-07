from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from django.shortcuts import get_object_or_404

from gestures.models import GestureSample


class LipInferenceView(APIView):
    """Mock lip inference endpoint.

    Accepts POST with either:
    - sample_uuid: run inference on stored sample
    - face_frames: list of landmark lists for frames

    Returns a JSON with predicted_text and note.
    """
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        sample_uuid = request.data.get('sample_uuid')
        face_frames = request.data.get('face_frames')

        # If sample provided, try to use its transcript as mock prediction
        if sample_uuid:
            sample = get_object_or_404(GestureSample, uuid=sample_uuid)
            if sample.transcript:
                return Response({'predicted_text': sample.transcript, 'note': 'Mock prediction (ground truth returned)'});
            else:
                return Response({'predicted_text': '', 'note': 'No transcript available for sample'}, status=status.HTTP_200_OK)

        # Else if face frames provided, run the demo decoder
        if face_frames:
            from .decoder import predict_from_face_frames
            pred = predict_from_face_frames(face_frames)
            return Response({'predicted_text': pred, 'note': f'Demo landmark-based decoder (heuristic) — processed {len(face_frames)} frames.'}, status=status.HTTP_200_OK)

        return Response({'error': 'No input provided'}, status=status.HTTP_400_BAD_REQUEST)
