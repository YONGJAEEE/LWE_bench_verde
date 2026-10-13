#!/usr/bin/env python3
"""Run the t-version benchmark generator and create Verde-ready datasets.

The benchmark generator writes one directory per assumed prefix bit
(`t_0` ... `t_7`).  This wrapper converts each selected directory immediately
to the Verde input layout, so training never has to consume the benchmark
layout directly.
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BUNDLE_ROOT = HERE.parent
GENERATOR = BUNDLE_ROOT / "benchmarking" / "src" / "generate" / "generate_A_b.py"
CHUNK = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate t-version benchmark data and convert it for Verde."
    )
    parser.add_argument(
        "--dump_path",
        required=True,
        type=Path,
        help="Benchmark generator output directory (contains t_0 ... t_7).",
    )
    parser.add_argument(
        "--verde_output_root",
        required=True,
        type=Path,
        help="Directory where Verde-ready t_i directories are created.",
    )
    parser.add_argument("--hamming", required=True, type=int)
    parser.add_argument("--secret_seed", type=int, default=0)
    parser.add_argument(
        "--t_version",
        type=int,
        default=None,
        help="Convert only this version (0..7); default converts every generated version.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing converted files to be replaced.",
    )
    parser.add_argument(
        "--convert_only",
        action="store_true",
        help="Skip generation and convert existing benchmark output.",
    )

    # Generator options needed for a normal run.  They are intentionally
    # explicit so the wrapper cannot accidentally pass conflicting values.
    parser.add_argument("--processed_dump_path", type=Path)
    parser.add_argument("--secret_path", type=Path, default=None)
    parser.add_argument("--num_secret_seeds", type=int, default=1)
    parser.add_argument("--secret_type", choices=["binary"], default="binary")
    parser.add_argument("--sigma", type=float, default=3)
    parser.add_argument("--max_samples", type=int, default=4_010_000)
    parser.add_argument("--std_threshold", type=float, default=1.0)
    parser.add_argument("--rlwe", type=int, default=0)
    parser.add_argument("--nu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--exp_name", default="verde_t_version")
    parser.add_argument("--exp_id", default="")
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def run_generator(args: argparse.Namespace) -> None:
    if args.processed_dump_path is None:
        raise ValueError("--processed_dump_path is required unless --convert_only is used")

    command = [
        sys.executable,
        str(GENERATOR),
        "--dump_path",
        str(args.dump_path),
        "--processed_dump_path",
        str(args.processed_dump_path),
        "--min_hamming",
        str(args.hamming),
        "--max_hamming",
        str(args.hamming),
        "--num_secret_seeds",
        str(args.num_secret_seeds),
        "--secret_type",
        args.secret_type,
        "--sigma",
        str(args.sigma),
        "--max_samples",
        str(args.max_samples),
        "--std_threshold",
        str(args.std_threshold),
        "--rlwe",
        str(args.rlwe),
        "--nu",
        str(args.nu),
        "--seed",
        str(args.seed),
        "--log_every",
        str(args.log_every),
        "--exp_name",
        args.exp_name,
        "--exp_id",
        args.exp_id,
        "--tag",
        args.tag,
        "--actions",
        "secrets",
    ]
    if args.secret_path is not None:
        command.extend(["--secret_path", str(args.secret_path)])

    print("Running benchmark generator:", " ".join(command))
    subprocess.run(command, cwd=BUNDLE_ROOT / "benchmarking", check=True)


def find_secret_dir(version_dir: Path, hamming: int, seed: int) -> Path:
    filename = f"train_b_{hamming}_{seed}.npy"
    matches = sorted(version_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} below benchmark output {version_dir}"
        )
    return matches[0].parent


def write_prefix(A: np.ndarray, b: np.ndarray, output: Path, chunk: int = CHUNK) -> None:
    total = len(A)
    if total == 0:
        raise ValueError(f"Cannot write an empty Verde prefix file: {output}")
    N = A.shape[1]
    fmt = ["%d"] * N + ["; %d"]
    with output.open("w", encoding="utf-8") as stream:
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            A_chunk = np.asarray(A[start:end], dtype=np.int64)
            b_chunk = np.asarray(b[start:end], dtype=np.int64).reshape(-1, 1)
            if len(A_chunk) != len(b_chunk):
                raise ValueError(f"A/b length mismatch while writing {output}")
            np.savetxt(
                stream,
                np.concatenate([A_chunk, b_chunk], axis=1),
                fmt=fmt,
                delimiter=" ",
            )
    print(f"Wrote {output} ({total:,} rows)")


def convert_version(
    version_dir: Path,
    output_dir: Path,
    hamming: int,
    secret_seed: int,
    overwrite: bool,
) -> None:
    secret_dir = find_secret_dir(version_dir, hamming, secret_seed)
    required = [
        version_dir / "train_A.npy",
        version_dir / "test_A.npy",
        version_dir / "orig_A.npy",
        secret_dir / f"train_b_{hamming}_{secret_seed}.npy",
        secret_dir / f"test_b_{hamming}_{secret_seed}.npy",
        secret_dir / f"orig_b_{hamming}_{secret_seed}.npy",
        secret_dir / f"secret_{hamming}_{secret_seed}.npy",
        secret_dir / "params.pkl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing benchmark files:\n" + "\n".join(missing))

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_A = np.load(required[0], mmap_mode="r")
    test_A = np.load(required[1], mmap_mode="r")
    orig_A = np.load(required[2], mmap_mode="r")
    train_b = np.load(required[3], mmap_mode="r").reshape(-1)
    test_b = np.load(required[4], mmap_mode="r").reshape(-1)
    orig_b = np.load(required[5], mmap_mode="r").reshape(-1, 1)
    secret = np.load(required[6], mmap_mode="r").reshape(-1, 1)

    if train_A.ndim != 2 or test_A.ndim != 2 or orig_A.ndim != 2:
        raise ValueError("Benchmark A files must be two-dimensional")
    if train_A.shape[1] != 80 or test_A.shape[1] != 80 or orig_A.shape[1] != 80:
        raise ValueError("t-version conversion requires N=80 (8+72)")
    if len(train_A) != len(train_b) or len(test_A) != len(test_b):
        raise ValueError("Benchmark A/b row counts do not match")
    if secret.shape[0] not in (72, 80):
        raise ValueError("Benchmark secret must contain 72 guesses or an 80-vector")

    np.save(output_dir / "orig_A.npy", np.asarray(orig_A))
    np.save(output_dir / "orig_b.npy", np.asarray(orig_b))
    np.save(output_dir / "secret.npy", np.asarray(secret))
    write_prefix(train_A, train_b, output_dir / "train.prefix")
    write_prefix(test_A, test_b, output_dir / "test.prefix")

    with required[7].open("rb") as stream:
        params = pickle.load(stream)
    if not isinstance(params, dict):
        params = vars(params)
    params = dict(params)
    params.update(
        {
            "N": 80,
            "secret_type": "binary",
            "min_hamming": hamming,
            "max_hamming": hamming,
            "num_secret_seeds": 1,
            "t_version": int(version_dir.name.split("_")[-1]),
        }
    )
    with (output_dir / "params.pkl").open("wb") as stream:
        pickle.dump(params, stream)

    print(
        f"Converted {version_dir.name}: train={len(train_A):,}, "
        f"test={len(test_A):,}, secret={secret.shape} -> {output_dir}"
    )


def main() -> None:
    args = parse_args()
    if args.t_version is not None and args.t_version not in range(8):
        raise ValueError("--t_version must be in the range 0..7")
    args.dump_path.mkdir(parents=True, exist_ok=True)
    args.verde_output_root.mkdir(parents=True, exist_ok=True)

    if not args.convert_only:
        run_generator(args)

    versions = [args.t_version] if args.t_version is not None else range(8)
    for version in versions:
        version_dir = args.dump_path / f"t_{version}"
        if not version_dir.is_dir():
            raise FileNotFoundError(f"Benchmark output not found: {version_dir}")
        convert_version(
            version_dir,
            args.verde_output_root / f"t_{version}",
            args.hamming,
            args.secret_seed,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
