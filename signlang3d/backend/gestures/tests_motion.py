from channels.testing import WebsocketCommunicator
from django.test import TransactionTestCase
from core.asgi import application
import asyncio

class InferenceConsumerMotionTest(TransactionTestCase):
    async def _run(self):
        comm = WebsocketCommunicator(application, '/ws/gesture/inference/', headers=[(b'origin', b'http://testserver')])
        connected, _ = await comm.connect()
        self.assertTrue(connected)

        # Create frames with almost no motion (same centroid)
        frame = {'hands':[{'landmarks':[{'x':0.5,'y':0.5,'z':0} for _ in range(21)]}], 'face':[{'x':0.5,'y':0.5,'z':0} for _ in range(468)]}
        frames = [frame for _ in range(30)]

        await comm.send_json_to({'type':'inference','request_id':'m1','model':'hybrid','language':'ASL','frames':frames})
        resp = await comm.receive_json_from()
        self.assertEqual(resp.get('type'),'error')
        self.assertIn('Low motion', resp.get('message',''))
        await comm.disconnect()

    def test_low_motion_rejected(self):
        return asyncio.run(self._run())
