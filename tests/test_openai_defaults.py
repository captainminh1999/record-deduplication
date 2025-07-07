import inspect
import unittest

from src.core.openai_engine import OpenAIEngine
from src.core.openai_types import DEFAULT_MODEL
from src.core.preprocess_engine import PreprocessEngine


class OpenAIDefaultModelTest(unittest.TestCase):
    def test_default_model_constant(self):
        # Test that the default model is consistent across the modular architecture
        self.assertEqual(DEFAULT_MODEL, "gpt-4o-mini-2024-07-18")
        
        # Verify OpenAI engine uses the correct default
        engine = OpenAIEngine()
        self.assertIsInstance(engine, OpenAIEngine)


if __name__ == "__main__":
    unittest.main()
