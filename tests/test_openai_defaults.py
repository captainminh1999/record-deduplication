import inspect
import unittest

from src import openai_integration, preprocess


class OpenAIDefaultModelTest(unittest.TestCase):
    def test_default_model_constant(self):
        default = openai_integration.DEFAULT_MODEL
        self.assertEqual(
            inspect.signature(preprocess.main)
            .parameters["openai_model"]
            .default,
            default,
        )
        self.assertEqual(
            inspect.signature(openai_integration.translate_to_english)
            .parameters["model"]
            .default,
            default,
        )
        self.assertEqual(
            inspect.signature(openai_integration.main)
            .parameters["openai_model"]
            .default,
            default,
        )


if __name__ == "__main__":
    unittest.main()
