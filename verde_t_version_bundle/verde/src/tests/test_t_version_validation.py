import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarking"))
sys.path.insert(1, str(ROOT / "verde"))

# The evaluator test is also importable as a src namespace package. If it ran
# first, clear its incompatible src.utils module before importing the
# benchmarking generator.
if "src.utils" in sys.modules and not hasattr(sys.modules["src.utils"], "create_this_logger"):
    del sys.modules["src.utils"]

from src.generate import generate_A_b as generator
from src.train.envs.datasets import EnvDataset


def make_dataset_params(root, secret, n=80, t_version=6, sigma=3.0):
    data_path = root / "data.txt"
    data_path.write_text(
        " ".join(["0"] * n) + "; 0\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        env_base_seed=0,
        batch_size=1,
        add_unred_perc=0,
        secret_col=0,
        data_cols=None,
        dense_cols=[],
        num_workers=0,
        global_rank=0,
        local_rank=0,
        n_gpu_per_node=1,
        reload_size=1,
        eval_size=1,
        train_path=str(data_path),
        test_path=str(data_path),
        batch_load=False,
        shuffle=False,
        Q=101,
        N=n,
        reload_data=str(root),
        secret=np.asarray(secret),
        t_version=t_version,
        sigma=sigma,
    )


class TestDataCheck(unittest.TestCase):
    def setUp(self):
        generator.params = SimpleNamespace(Q=97, write_threshold=2.0)

    def test_single_secret_checks_good_residual_and_accepts(self):
        rng = np.random.default_rng(7)
        ra = rng.integers(0, 97, size=(32, 4), dtype=np.int64)
        secret = np.array([1, 0, 1, 0], dtype=np.int64)
        rb = (ra @ secret) % 97
        self.assertTrue(generator.data_check(ra, rb, secret))
        self.assertEqual(generator.data_check.last_result["reason"], "accepted")

    def test_structural_and_modular_validation_rejects_bad_batches(self):
        ra = np.zeros((4, 4), dtype=np.int64)
        secret = np.zeros(4, dtype=np.int64)
        rb = np.zeros(4, dtype=np.int64)
        self.assertFalse(generator.data_check(ra, rb, secret.astype(float)))
        self.assertEqual(
            generator.data_check.last_result["reason"], "secret_not_integer"
        )
        self.assertFalse(
            generator.data_check(ra, np.array([0, 0, 0, 97]), secret)
        )
        self.assertEqual(
            generator.data_check.last_result["reason"],
            "modular_value_out_of_range",
        )


class TestEnvDatasetSecretValidation(unittest.TestCase):
    def _write_original_data(self, root, secret, q=101, corrupt=False):
        rng = np.random.default_rng(9)
        matrix = rng.integers(0, q, size=(16, 80), dtype=np.int64)
        b = (matrix @ secret) % q
        if corrupt:
            b[0] = (b[0] + 40) % q
        np.save(root / "orig_A.npy", matrix)
        np.save(root / "orig_b.npy", b)

    def test_t6_validates_correct_secret_and_wrong_t_is_negative_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            secret = np.zeros(80, dtype=np.int64)
            secret[6] = 1
            secret[17] = 1
            self._write_original_data(root, secret)

            correct = make_dataset_params(root, secret[8:], t_version=6)
            dataset = EnvDataset(correct, None, "test")
            np.testing.assert_array_equal(dataset.orig_A.shape, (16, 80))

            wrong = make_dataset_params(root, secret[8:], t_version=0)
            EnvDataset(wrong, None, "test")

    def test_t6_rejects_incorrect_residual(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            secret = np.zeros(80, dtype=np.int64)
            secret[6] = 1
            self._write_original_data(root, secret, corrupt=True)
            params = make_dataset_params(root, secret[8:], t_version=6)
            with self.assertRaisesRegex(ValueError, "t-version 6 residual validation failed"):
                EnvDataset(params, None, "test")

    def test_legacy_validation_remains_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(11)
            secret = np.array([1, 0, 1, 0], dtype=np.int64)
            matrix = rng.integers(0, 101, size=(12, 4), dtype=np.int64)
            np.save(root / "orig_A.npy", matrix)
            np.save(root / "orig_b.npy", (matrix @ secret) % 101)
            params = make_dataset_params(root, secret, n=4, t_version=-1)
            dataset = EnvDataset(params, None, "test")
            np.testing.assert_array_equal(dataset.orig_A, matrix)


if __name__ == "__main__":
    unittest.main()
