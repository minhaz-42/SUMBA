from django.test import TestCase, Client
from django.urls import reverse
import json

class LipInferenceTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_lip_inference_returns_prediction_for_face_frames(self):
        url = reverse('lip_inference')
        # synthetic mouth landmarks (indices not important here)
        face_frame1 = [{'x':0.5,'y':0.45,'z':0.0} for _ in range(12)]
        face_frame2 = [{'x':0.5,'y':0.55,'z':0.0} for _ in range(12)]
        resp = self.client.post(url, data=json.dumps({'face_frames':[face_frame1, face_frame2]}), content_type='application/json')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('predicted_text', data)
        self.assertTrue(data['predicted_text'] == '' or isinstance(data['predicted_text'], str))
