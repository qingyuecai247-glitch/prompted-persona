"""Offline service/configuration checks; never invoke inference."""
from pathlib import Path
import importlib.util
import os
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

def load_generator():
    spec = importlib.util.spec_from_file_location('response_generator', ROOT / '02_behavior_generation/layers-llm.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {'PERSONA_MODEL_SOURCE':'local','PERSONA_LLM_URL':'','AZURE_OPENAI_ENDPOINT':'','AZURE_OPENAI_API_KEY':''}):
        spec.loader.exec_module(module)
    return module

class GenerationConfiguration(unittest.TestCase):
    def test_missing_local_endpoint_fails_before_request(self):
        module = load_generator()
        with patch.object(module.requests, 'post') as post:
            with self.assertRaisesRegex(RuntimeError, 'PERSONA_LLM_URL'):
                module.call_llm('synthetic prompt')
            post.assert_not_called()

    def test_custom_service_json_string_is_accepted(self):
        module = load_generator()
        module.LOCAL_LLM_URL = 'synthetic-test-endpoint'
        response = unittest.mock.Mock(status_code=200)
        response.json.return_value = '{"synthetic_score": 4}'
        with patch.object(module.requests, 'post', return_value=response) as post:
            text, prompt_tokens, completion_tokens = module.call_llm('synthetic prompt')
        self.assertEqual(text, '{"synthetic_score": 4}')
        self.assertEqual((prompt_tokens, completion_tokens), (0, 0))
        self.assertEqual(post.call_args.args[0], 'synthetic-test-endpoint')

    def test_chat_completion_object_is_accepted(self):
        module = load_generator()
        module.LOCAL_LLM_URL = 'synthetic-test-endpoint'
        response = unittest.mock.Mock(status_code=200)
        response.json.return_value = {'choices':[{'message':{'content':' synthetic response '}}]}
        with patch.object(module.requests, 'post', return_value=response):
            text, _, _ = module.call_llm('synthetic prompt')
        self.assertEqual(text, 'synthetic response')

if __name__ == '__main__':
    unittest.main()
