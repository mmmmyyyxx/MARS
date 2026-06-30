import unittest

from mars_utils.evaluator import (
    canonical_answer,
    compute_final_metrics_from_predictions,
    is_valid_canonical,
)


class EvaluatorTest(unittest.TestCase):
    def test_canonical_answer_formats(self):
        self.assertEqual(canonical_answer("The answer is True.", "boolean"), "true")
        self.assertEqual(canonical_answer("False", "boolean"), "false")
        self.assertEqual(canonical_answer("invalid", "valid_invalid"), "invalid")
        self.assertEqual(canonical_answer("valid", "valid_invalid"), "valid")
        self.assertEqual(canonical_answer("yes.", "yes_no"), "yes")
        self.assertEqual(canonical_answer("(C)", "option_letter"), "(C)")

    def test_invalid_is_not_parsed_as_valid(self):
        self.assertEqual(
            canonical_answer("This argument is invalid.", "valid_invalid"), "invalid"
        )

    def test_option_letter_is_only_valid_for_option_format(self):
        self.assertFalse(is_valid_canonical("(A)", "yes_no"))
        self.assertTrue(is_valid_canonical("(A)", "option_letter"))

    def test_final_metrics_use_last_iteration_predictions(self):
        predictions = [
            {"iteration": 1, "correct": True},
            {"iteration": 1, "correct": False},
            {"iteration": 2, "correct": True},
            {"iteration": 2, "correct": True},
            {"iteration": 2, "correct": False},
        ]
        metrics = compute_final_metrics_from_predictions(predictions)
        self.assertEqual(metrics["num_samples"], 3)
        self.assertEqual(metrics["num_success"], 2)
        self.assertEqual(metrics["num_failed"], 1)
        self.assertAlmostEqual(metrics["final_accuracy"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
