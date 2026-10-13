#!/usr/bin/env python3
"""Run the t-version benchmark generator and create Verde-ready datasets.

The benchmark generator writes one directory per assumed prefix bit
(`t_0` ... `t_7`).  This wrapper converts each selected directory immediately
to the Verde input layout, so training never has to consume the benchmark
layout directly.
"""

from __future__ import annotations

import argparse
import os
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
    parser.add_argument(
        "--orig_A_path",
        type=Path,
        default=None,
        help="Optional explicit original-A path; otherwise resolve it from params.pkl.",
    )
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


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def ensure_benchmark_tree_link(processed_dump_path: Path) -> None:
    """Make the legacy relative benchmark path point at this run's input."""
    link = (
        BUNDLE_ROOT
        / "benchmarking"
        / "data"
        / "benchmark_paper_data"
        / "n80_logq7"
    )
    link.parent.mkdir(parents=True, exist_ok=True)
    target = processed_dump_path.resolve()
    if link.is_symlink():
        if link.resolve() != target:
            print(
                f"Benchmark compatibility link already points to {link.resolve()}, "
                f"leaving it unchanged (using explicit absolute paths for this run)."
            )
        return
    if link.exists():
        print(
            f"Benchmark compatibility path exists at {link}; leaving it unchanged "
            "(using explicit absolute paths for this run)."
        )
        return
    link.symlink_to(target, target_is_directory=True)
    print(f"Created benchmark compatibility link: {link} -> {target}")


def load_processed_params(processed_dump_path: Path) -> dict:
    candidates = [processed_dump_path / "params.pkl"]
    candidates.extend(sorted(processed_dump_path.rglob("params.pkl")))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        with candidate.open("rb") as stream:
            params = pickle.load(stream)
        return dict(params) if isinstance(params, dict) else vars(params)
    raise FileNotFoundError(f"No params.pkl found below {processed_dump_path}")


def resolve_original_A_path(args: argparse.Namespace) -> Path:
    if args.orig_A_path is not None:
        path = resolve_path(args.orig_A_path)
        if not path.is_file():
            raise FileNotFoundError(f"--orig_A_path does not exist: {path}")
        return path

    processed = resolve_path(args.processed_dump_path)
    params = load_processed_params(processed)
    raw = params.get("reload_data", params.get("orig_A_path", ""))
    if not raw:
        raise ValueError(
            f"params.pkl below {processed} does not define reload_data or orig_A_path"
        )
    raw_path = Path(os.path.expanduser(str(raw)))
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                Path.cwd() / raw_path,
                processed / raw_path,
                processed.parent / raw_path,
                BUNDLE_ROOT / raw_path,
                BUNDLE_ROOT / "benchmarking" / raw_path,
                processed / raw_path.name,
            ]
        )
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not resolve original A matrix from params.pkl. "
        f"Configured path={raw!r}; checked:\n" + "\n".join(str(p) for p in candidates)
    )


def run_generator(args: argparse.Namespace) -> None:
    if args.processed_dump_path is None:
        raise ValueError("--processed_dump_path is required unless --convert_only is used")

    python_bin = os.environ.get("PYTHON") or shutil.which("python") or sys.executable
    orig_A_path = resolve_original_A_path(args)
    command = [
        python_bin,
        str(GENERATOR),
        "--dump_path",
        str(args.dump_path),
        "--processed_dump_path",
        str(args.processed_dump_path),
        "--orig_A_path",
        str(orig_A_path),
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

    env = os.environ.copy()
    env["BUNDLE"] = str(BUNDLE_ROOT)
    env["PYTHON"] = python_bin
    print("BUNDLE:", env["BUNDLE"])
    print("PYTHON:", env["PYTHON"])
    print("Running benchmark generator:", " ".join(command))
    subprocess.run(command, cwd=BUNDLE_ROOT / "benchmarking", env=env, check=True)


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
    orig_b = np.asarray(np.load(required[5], mmap_mode="r"))
    secret = np.asarray(np.load(required[6], mmap_mode="r"))

    if train_A.ndim != 2 or test_A.ndim != 2 or orig_A.ndim != 2:
        raise ValueError("Benchmark A files must be two-dimensional")
    if train_A.shape[1] != 80 or test_A.shape[1] != 80 or orig_A.shape[1] != 80:
        raise ValueError("t-version conversion requires N=80 (8+72)")
    if len(train_A) != len(train_b) or len(test_A) != len(test_b):
        raise ValueError("Benchmark A/b row counts do not match")
    if orig_b.ndim == 1:
        orig_b = orig_b.reshape(-1, 1)
    if orig_b.ndim != 2 or orig_b.shape[1] <= secret_seed:
        raise ValueError(
            f"Benchmark orig_b must have a secret column for seed {secret_seed}; "
            f"got shape {orig_b.shape}"
        )
    if secret.ndim == 1:
        secret = secret.reshape(-1, 1)
    if secret.shape[0] == 72:
        padded_secret = np.zeros((80, secret.shape[1]), dtype=secret.dtype)
        padded_secret[8:, :] = secret
        secret = padded_secret
    if secret.ndim != 2 or secret.shape[0] != 80:
        raise ValueError("Benchmark secret must contain a padded 80-vector")
    if secret.shape[1] <= secret_seed:
        raise ValueError(
            f"Benchmark secret must have a column for seed {secret_seed}; "
            f"got shape {secret.shape}"
        )
    s1_weight = int(np.count_nonzero(secret[8:, secret_seed]))
    if s1_weight != hamming:
        raise ValueError(
            f"Benchmark secret s_1 hamming mismatch: expected {hamming}, got {s1_weight}"
        )
    if np.count_nonzero(secret[:8, secret_seed]) != 0:
        raise ValueError("Benchmark padded secret must have a zero t prefix")

    np.save(output_dir / "orig_A.npy", np.asarray(orig_A))
    np.save(output_dir / "orig_b.npy", np.asarray(orig_b[:, secret_seed : secret_seed + 1]))
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
    if args.num_secret_seeds != 1:
        raise ValueError("This Verde t-version bundle requires --num_secret_seeds 1")
    if args.secret_seed != 0:
        raise ValueError("This Verde t-version bundle requires --secret_seed 0")
    args.dump_path = resolve_path(args.dump_path)
    args.verde_output_root = resolve_path(args.verde_output_root)
    if args.processed_dump_path is not None:
        args.processed_dump_path = resolve_path(args.processed_dump_path)
    if args.orig_A_path is not None:
        args.orig_A_path = resolve_path(args.orig_A_path)
    args.dump_path.mkdir(parents=True, exist_ok=True)
    args.verde_output_root.mkdir(parents=True, exist_ok=True)

    if not args.convert_only:
        if args.processed_dump_path is None:
            raise ValueError("--processed_dump_path is required unless --convert_only is used")
        ensure_benchmark_tree_link(args.processed_dump_path)
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
