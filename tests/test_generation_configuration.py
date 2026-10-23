"""Offline service/configuration checks; never invoke inference."""
from pathlib import Path
import importlib.util
import os
import unittest
from unittest.mock import patch

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "config/paths.json").is_file())

def load_generator():
    spec = importlib.util.spec_from_file_location('response_generator', ROOT / 'generation/responses.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, {'PERSONA_MODEL_SOURCE':'local','PERSONA_LLM_URL':'','AZURE_OPENAI_ENDPOINT':'','AZURE_OPENAI_API_KEY':''}):
        spec.loader.exec_module(module)
    return module

class GenerationConfiguration(unittest.TestCase):
    def test_existing_generator_uses_late_items_in_full_scoring(self):
        module = load_generator()
        answers = {f"Q{i}": "Neither Accurate Nor Inaccurate" for i in range(1, 61)}
        answers["Q60"] = "Very Accurate"
        answers["Q58"] = "Very Accurate"
        scores = module.aggregate_ocean(module.map_big5_scores(answers))
        self.assertAlmostEqual(scores["Openness"], 3 + 2/12)
        self.assertAlmostEqual(scores["Conscientiousness"], 3 - 2/12)

    def test_existing_generator_rejects_incomplete_baseline(self):
        module = load_generator()
        answers = {f"Q{i}": 3 for i in range(1, 60)}
        with self.assertRaisesRegex(ValueError, "Q60"):
            module.aggregate_ocean(answers)

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
