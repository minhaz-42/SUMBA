from channels.testing import WebsocketCommunicator
from django.test import TransactionTestCase
from django.conf import settings
from core.asgi import application
import asyncio

class InferenceConsumerEmptyFramesTest(TransactionTestCase):
    async def _run(self):
        comm = WebsocketCommunicator(application, '/ws/gesture/inference/', headers=[(b'origin', b'http://testserver')])
        connected, _ = await comm.connect()
        self.assertTrue(connected)

        # Send a batch with empty frames
        await comm.send_json_to({'type': 'inference', 'request_id': 'test1', 'model': 'hybrid', 'language': 'ASL', 'frames': []})
        response = await comm.receive_json_from()
        self.assertEqual(response.get('type'), 'error')
        await comm.disconnect()

    def test_empty_frames_rejected(self):
        return asyncio.run(self._run())
