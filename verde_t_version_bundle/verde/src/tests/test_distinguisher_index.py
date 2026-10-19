import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "verde"))

from src.train.evaluator import Evaluator


class RecordingSecretCheck:
    valid_indices = np.arange(8, 80, dtype=int)

    def __init__(self):
        self.logs = {}

    def add_log(self, name, value):
        self.logs[name] = value

    def scores_to_full(self, scores):
        return np.asarray(scores).reshape(-1)

    def match_secret_iter(self, indices, scores, method):
        return False


class TestDistinguisherIndexing(unittest.TestCase):
    def test_perturbation_predictions_are_stored_by_coordinate(self):
        evaluator = object.__new__(Evaluator)
        evaluator.params = SimpleNamespace(
            N=80, Q=101, distinguisher_size=4, secret_type="binary"
        )
        evaluator.secret_check = RecordingSecretCheck()
        evaluator.iterator = SimpleNamespace(
            dataset=SimpleNamespace(
                getbatchA=lambda n: np.zeros((n, 80), dtype=np.int64)
            )
        )

        def mock_predict(matrix, encoder, decoder, intermediate=False):
            changed = np.flatnonzero(np.any(matrix != 0, axis=0))
            value = 0 if len(changed) == 0 else float(changed[0])
            return np.full(len(matrix), value)

        evaluator.predict_outputs = mock_predict
        evaluator.run_distinguisher(None, None)

        predictions = evaluator.secret_check.logs["distinguisher_bits"]
        self.assertEqual(predictions[0][0], 0)
        self.assertEqual(predictions[7][0], 0)
        self.assertEqual(predictions[8][0], 8)
        self.assertEqual(predictions[9][0], 9)
        self.assertEqual(predictions[79][0], 79)
        self.assertGreater(evaluator.distinguisher_results[8], 0)
        self.assertNotEqual(
            evaluator.distinguisher_results[8], evaluator.distinguisher_results[9]
        )


if __name__ == "__main__":
    unittest.main()
