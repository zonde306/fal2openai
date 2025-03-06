import os
import unittest
import app
import features
import fal
import defines
import blacksheep
import blacksheep.testing

class TestApp(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await app.app.start()
        self.client = blacksheep.testing.TestClient(app.app)
    
    async def test_models(self):
        response = await self.client.get("/v1/models")
        data : dict = await response.json()
        self.assertEqual(response.status, 200)
        for model in data.get("data", []):
            self.assertEqual(model["object"], "model")
            self.assertIn("id", model)
            self.assertIn(model.get("name", ""), defines.MODELS)
    
    async def test_chat_completions(self):
        key = os.getenv("TRUE_KEY")
        messages = [
            {
                "role": "system",
                "content": "you are helpful assistant"
            },
            {
                "role": "user",
                "content": "hi",
            },
        ]
        print("---")
        async for msg in fal.send_message(messages, key, "deepseek/deepseek-r1"):
            print(msg["choices"][0]["delta"]["content"], end="")
            if msg["choices"][0]["finish_reason"] == "completed":
                print("")
                self.assertEqual(msg["choices"][0]["finish_reason"], "completed")
            else:
                self.assertIsNone(msg["choices"][0]["finish_reason"])
        
        messages = [
            {
                "role": "system",
                "content": "you are helpful assistant"
            },
            {
                "role": "user",
                "content": "hi",
            },
        ]
        print("---")
        msg = await fal.send_message_sync(messages, key, "deepseek/deepseek-r1")
        print(msg)
        self.assertEqual(msg["choices"][0]["finish_reason"], "completed")
        self.assertNotIn("ERROR", msg["choices"][0]["message"]["content"])



class TestFeatures(unittest.IsolatedAsyncioTestCase):
    def test_features(self):
        messages = [{
            "role": "system",
            "content": "first message\n"
                       "<roleInfo>\n"
                       "user:asdf8249\n"
                       "assistant: asfdf\n"
                       "system:fwasd\n"
                       "developer: ddd\n"
                       "</roleInfo>\n"
                       "just system message"
        }]
        feat = features.process_features(messages)

        self.assertEqual(feat.ROLE.user, "asdf8249")
        self.assertEqual(feat.ROLE.assistant, "asfdf")
        self.assertEqual(feat.ROLE.system, "fwasd")
        self.assertEqual(feat.ROLE.developer, "ddd")
        self.assertIn("just system", feat.SYSTEM_PROMPT)

class TestFal(unittest.IsolatedAsyncioTestCase):
    async def test_format_messages(self):
        messages = [
            {
                "role": "system",
                "content": "<roleInfo>\n"
                           "user:asdf8249\n"
                           "assistant: asfdf\n"
                           "system:fwasd\n"
                           "developer: ddd\n"
                           "</roleInfo>\njust system"
            },
            {
                "role": "user",
                "content": "hi",
            },
        ]
        feat = features.process_features(messages)
        prompt, _ = await fal.format_messages(messages, feat.ROLE)

        self.assertNotIn("user: hi", prompt)
        self.assertIn("asdf8249: hi", prompt)
    
    async def test_error(self):
        messages = [
            {
                "role": "system",
                "content": "you are helpful assistant"
            },
            {
                "role": "user",
                "content": "hi",
            },
        ]
        async for msg in fal.send_message(messages, "", "openai/gpt-4o-mini"):
            print(msg)
            self.assertEqual(msg["choices"][0]["finish_reason"], "error")
            self.assertIn("ERROR", msg["choices"][0]["delta"]["content"])
        
        messages = [
            {
                "role": "system",
                "content": "you are helpful assistant"
            },
            {
                "role": "user",
                "content": "hi",
            },
        ]
        msg = await fal.send_message_sync(messages, "", "openai/gpt-4o-mini")
        print(msg)
        self.assertEqual(msg["choices"][0]["finish_reason"], "error")
        self.assertIn("ERROR", msg["choices"][0]["message"]["content"])
    