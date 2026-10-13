"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Very simple script for generating secret datasets. It only supports LWE (no ring LWE) and does not support Cathy's "storage efficient" option.

The main arguments:

--processed_dump_path: the directory with data_*.prefix files, or the parent of that directory. See the NOTE in the comments.
--secret_path: path to an existing secret.npy file. If not provided, the script will generate a new secret.
--secret_type: binary, ternary, gaussian, binomial. Look at the SecretFactory classes.
--actions: one or more of "secrets", "only_secrets", "plot", "describe"
    * secrets: given A data, generate new secrets using --min_hamming, --max_hamming, etc., and associated b data.
    * only_secrets: given N, min_hamming, max_hamming, and secret type, generate only the secret.npy file (useful for uSVP and MiTM attacks).
    * plot: plot reduction levels for RA and save to the output directory.
    * describe: calculate a lot of statistics about the generated data.

python generate_A_b.py --processed_dump_path /path/to/preprocessed/data/ --exp_name debug --dump_path /path/to/dump/data --secret_type binary --num_secret_seeds 10 --min_hamming 1 --max_hamming 20 --actions secrets plot describe

This will make some binary secrets with hamming weights from 1 to 20 and describe the outputs.

Number of samples: 1350977
Regular std:  1203.9386
Centered std: 737.9868
Regular reduction:  1.2530
Centered reduction: 0.7677
Cruel bits:  143
Recommended max hamming: 10, max brute force bits: 6 , choices: 10.7B
BKZ block sizes: 35, 40
Std deviation of (RA*s - Rb mod Q) / Q: 0.322 (0.002)


You want to make sure that 

* the number of samples is what you expect (~4M)
* Centered reduction is similar to the write threshold of the preprocessing job
* BKZ block sizes don't surprise you (read from the original params file)

While reading from the data.prefix file, if there are issues in it (likely from concurrent writing to data.prefix), the logs will complain:

It might complain that the err std is too close to random. This means that (RA @ secret - Rb) is very close to a uniform error distribution, which is bad news.

It might also complain that the reduction is not below the original write threshold. This indicates a bug because the whole point of preprocessing is to get std(centered(RA)) under that threshold.
"""

import argparse
import gc
import getpass
import glob
import itertools
import json
import logging
import math
import os
import pickle
import shutil
import sys
import time
import numpy as np
from tqdm import tqdm
sys.path.append(".")
from src.generate.lllbkz import get_mlwe_circ, centered
from src.utils import create_this_logger, human, init_rng, mod_mult_torch, read, remove_redundant_rows, shift_negate, shuffled


def get_params():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(
        description="Generate secrets from reduced A matrices."
    )

    user = getpass.getuser()
    parser.add_argument("--dump_path", default=f"/checkpoint/{user}/dumped")
    parser.add_argument(
        "--exp_name", default="secrets", help="Experiment name"
    )
    parser.add_argument("--exp_id", default="", help="Experiment ID")
    parser.add_argument(
        "--tag", default="", help="Extra info to include in folder name."
    )
    parser.add_argument(
        "--log_every", default=500, type=int, help="How often to log progress."
    )

    parser.add_argument("--seed", type=int, default=42, help="-1 to use timestamp seed")
    parser.add_argument(
        "--actions",
        nargs="+",
        choices=["secrets", "plot", "describe", "only_secrets"],
        default=["secrets"],
        help="Which functions to run.",
    )

    # Load data
    parser.add_argument(
        "--processed_dump_path",
        required=True,
        help="Directory to load the preprocessed data from.",
    )
    parser.add_argument(
        "--orig_A_path",
        default="",
        help=(
            "Explicit path to the original A matrix. This avoids relying on "
            "the working directory when params.pkl stores a relative path."
        ),
    )
    parser.add_argument("--secret_path", default="", help="Path to predefined secret.npy file.")

    # LWE
    parser.add_argument("--num_secret_seeds", type=int, default=10)
    parser.add_argument("--N", type=int, default=512, help="LWE dimension for only_secrets")
    parser.add_argument("--min_hamming", type=int, default=3, help="Min hamming weight")
    parser.add_argument(
        "--max_hamming", type=int, default=20, help="Max hamming weight."
    )
    parser.add_argument(
        "--secret_type",
        default="binary",
        choices=["binary", "ternary", "gaussian", "binomial"],
    )
    parser.add_argument(
        "--sigma", type=float, default=3, help="sigma for gaussian error"
    )
    parser.add_argument("--gamma", type=int, default=2, help="for binomial error")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=4_010_000,
        help="Maximum number of training samples. The number of training samples might be less than this number.",
    )
    parser.add_argument(
        "--std_threshold",
        default=1.0,
        type=float,
        help="Minimum std deviation of reduced A to use.",
    )

    # RLWE
    parser.add_argument("--rlwe", type=int, default=0, help="LWE (0) or RLWE (1), MLWE (k>1)")
    parser.add_argument("--nu", type=int, default=0, help="Number of cruel bits, used to split the cliff, if provided.")

    params = parser.parse_args()

    # Save command used
    params.command = (" ".join(["python"] + sys.argv),)

    return params


# Secret types
class BinaryFactory:
    type_ = "binary"

    def __init__(self, *, sigma):
        self.sigma = sigma

    def new(self, N, min_h, max_h, num_secret_seeds):
        num_secrets = num_secret_seeds * (max_h - min_h + 1)
        full = self._fill(shape=(N, num_secrets))
        secret = self._sparsify(full, min_h, max_h, num_secret_seeds)
        return secret

    def _fill(self, shape):
        return np.ones(shape, dtype=np.int64)

    def _sparsify(self, full, min_h, max_h, num_secret_seeds):
        for h in range(min_h, max_h + 1):
            for seed_id in range(num_secret_seeds):
                column = (h - min_h) * num_secret_seeds + seed_id
                nonzeros = np.flatnonzero(full[:, column])

                if len(nonzeros) > h:
                    # Too many nonzero coordinates, randomly zero some out.
                    extra = len(nonzeros) - h
                    idxs = rng.choice(nonzeros, size=extra, replace=False)
                    full[idxs, column] = 0

                # Required Hamming weight is higher than nonzeros in secret.
                assert h >= np.count_nonzero(full[:, column])

        return full

    def error(self, shape):
        return rng.normal(0, self.sigma, size=shape).round().astype(np.int64)

    def __repr__(self):
        variables = ", ".join([f"{k}={v}" for k, v in vars(self).items()])
        return f"{self.__class__.__name__}({variables})"


class TernaryFactory(BinaryFactory):
    type_ = "ternary"

    def _fill(self, shape):
        return rng.choice([-1, 1], size=shape)


class GaussianFactory(BinaryFactory):
    type_ = "gaussian"

    def _fill(self, shape):
        decimals = rng.normal(0, self.sigma, size=shape)
        return decimals.round().astype(np.int64)


class BinomialFactory(BinaryFactory):
    type_ = "binomial"

    def __init__(self, *, gamma):
        self.gamma = gamma

    def _fill(self, shape):
        half1 = rng.binomial(self.gamma, 0.5, size=shape)
        half2 = rng.binomial(self.gamma, 0.5, size=shape)
        return half1 - half2

    def error(self, shape):
        err1 = rng.binomial(self.gamma, 0.5, shape)
        err2 = rng.binomial(self.gamma, 0.5, shape)
        return err1 - err2


def circulate_mlwe_A(A, n, k):
    new_mat = np.flip(A.reshape((len(A), k, n)), axis=2)
    circulated_mat = np.zeros((n, len(A), k, n), dtype=np.int64)
    for i in range(n):
        circulated_mat[~i] = new_mat
        new_mat = shift_negate(new_mat)

    circulated_mat = circulated_mat.swapaxes(0, 1)
    A = circulated_mat.reshape((-1, k * n))
    return A

def data_check(batch_RA, batch_Rb, secret, rlwe=0):
    """Will return False if it finds that stdev is bad"""
    # Skip file if (RA @ s - Rb) % Q is not significantly better.
    if (
        rlwe
    ):  # because indexing is different for RLWE, must generate circulant matrices to check.
        N, k = batch_RA.shape[1], rlwe
        n = N // k
        batch_RA = circulate_mlwe_A(batch_RA, n, k) % params.Q
        batch_Rb = batch_Rb.reshape((-1, batch_Rb.shape[-1]))

    bad_err_std = (
        centered((batch_RA @ secret[:, 1:] - batch_Rb[:, :-1]) % params.Q, params.Q).std()
        / params.Q
    )
    good_err_std = (
        centered((batch_RA @ secret - batch_Rb) % params.Q, params.Q).std() / params.Q
    )
    logger.info("Good err std: %f \t Bad err std: %f", good_err_std, bad_err_std)

    if good_err_std + 0.005 >= bad_err_std:
        logger.warning(
            "Err std too close to random. [err: %.3f, random: %.3f]",
            good_err_std,
            bad_err_std,
        )
        return False

    # skip file if stddev is above write threshold.
    std = np.sqrt(12) * centered(batch_RA, params.Q).std() / params.Q
    # logger.warning('RA std is %.3f', std)
    if std > params.write_threshold:
        logger.warning("RA above reduction threshold. [std: %.3f]", std)
        return False
    return True


def init_tiny_A(nu=0, k=0):
    logger.info(f"Loading from {params.orig_A_path}")
    A = np.load(params.orig_A_path, allow_pickle=True)
    N = A.shape[1]
    idxs = np.arange(N, dtype=int)
    if nu>0 and k>1:
        nu_parts = np.array_split(np.arange(nu), k)  # Split nu into k parts
        nr_parts = np.array_split(nu+np.arange(N-nu), k)  # Split nr into k parts

        idxs = []
        for i in range(k):
            idxs.extend(nu_parts[i])
            idxs.extend(nr_parts[i])

    A = A[:,idxs]
    return A, idxs


def generate(test_size=10000):
    if params.N != 80:
        raise ValueError(
            "t-version generation requires N=80 (8-bit t prefix + 72-bit guess); "
            f"got N={params.N}."
        )
    if not 0 <= params.min_hamming <= params.max_hamming <= 72:
        raise ValueError(
            "t-version hamming range must satisfy 0 <= min_hamming <= "
            "max_hamming <= 72; "
            f"got {params.min_hamming}..{params.max_hamming}."
        )
    if params.secret_type != "binary":
        raise ValueError(
            "t-version generation currently supports only binary secrets; "
            f"got secret_type={params.secret_type!r}."
        )

    if params.secret_type == "binary":
        secret_factory = BinaryFactory(sigma=params.sigma)
    elif params.secret_type == "ternary":
        secret_factory = TernaryFactory(sigma=params.sigma)
    elif params.secret_type == "gaussian":
        secret_factory = GaussianFactory(sigma=params.sigma)
    elif params.secret_type == "binomial":
        secret_factory = BinomialFactory(gamma=params.gamma)
    else:
        raise ValueError(params.secret_type)

    h_range = range(params.min_hamming, params.max_hamming + 1)
    num_secrets = len(h_range) * params.num_secret_seeds

    # t-version secrets are constructed directly in the split used by the
    # benchmark: t is fixed at index 6 and s_1 has exactly h nonzero entries
    # in indices 8..79.  Generating an 80-dimensional hamming-h secret first
    # and then deleting its prefix would make the effective s_1 weight random.
    if not os.path.isfile(params.secret_path):
        secret = np.zeros((params.N, num_secrets), dtype=np.int64)
        guess_indices = np.arange(8, params.N)
        for h in h_range:
            for seed_id in range(params.num_secret_seeds):
                column = (h - params.min_hamming) * params.num_secret_seeds + seed_id
                chosen = rng.choice(guess_indices, size=h, replace=False)
                secret[chosen, column] = 1
        secret[6, :] = 1
        logger.info(
            "Created t-version secrets with t[6]=1 and exact s_1 weights "
            "for h=%d..%d",
            params.min_hamming,
            params.max_hamming,
        )
    else:
        logger.info("Loading secret from %s", params.secret_path)
        loaded_secret = np.asarray(np.load(params.secret_path), dtype=np.int64)
        if loaded_secret.ndim == 1:
            loaded_secret = loaded_secret[:, None]
        if loaded_secret.ndim != 2 or loaded_secret.shape[0] not in (72, 80):
            raise ValueError(
                "t-version secret_path must contain a 72- or 80-row array; "
                f"got shape {loaded_secret.shape}."
            )
        if loaded_secret.shape[0] == 72:
            secret = np.zeros((80, loaded_secret.shape[1]), dtype=np.int64)
            secret[8:, :] = loaded_secret
        else:
            secret = loaded_secret.copy()
        if secret.shape[1] == 1 and num_secrets > 1:
            secret = np.repeat(secret, num_secrets, axis=1)
        if secret.shape != (80, num_secrets):
            raise ValueError(
                "t-version secret_path has the wrong number of secrets; "
                f"expected {(80, num_secrets)}, got {secret.shape}."
            )
        secret[:8, :] = 0
        secret[6, :] = 1

    s1_weights = np.count_nonzero(secret[8:, :], axis=0)
    expected_weights = np.repeat(
        np.arange(params.min_hamming, params.max_hamming + 1),
        params.num_secret_seeds,
    )
    if np.any((s1_weights < 0) | (s1_weights > 72)):
        raise ValueError(
            "s_1 hamming weight must be in the range 0..72; "
            f"got {s1_weights.tolist()}."
        )
    if not np.array_equal(s1_weights, expected_weights):
        raise ValueError(
            "t-version secret construction produced unexpected s_1 hamming "
            f"weights: expected {expected_weights.tolist()}, got {s1_weights.tolist()}."
        )

    # s를 s_0 (8행)와 s_1 (72행)로 분리
    # s는 80행 num_secrets열 형태 (N=80, num_secrets=20)
    s_0 = secret[:8, :]  # (8, num_secrets)
    s_1 = secret[8:80, :]  # (72, num_secrets)
    logger.info("Secret split into s_0 (8x%d) and s_1 (72x%d)", num_secrets, num_secrets)
    logger.info("s_0: %s (should have 1 at index 6, t_6)", s_0[:, 0])
    logger.info("s_1 hamming weight: %d", int(np.count_nonzero(s_1[:, 0])))
    logger.info("full secret hamming weight: %d", int(np.count_nonzero(secret[:, 0])))

    tiny_A, col_permutation = init_tiny_A(nu = params.nu, k=params.rlwe)

    # A 행렬을 A_0 (s_0에 대응하는 처음 8열)와 A_1 (s_1에 대응하는 마지막 72열)로 분리
    # tiny_A shape: (num_samples, N=80)
    tiny_A_0 = tiny_A[:, :8]  # s_0에 대응 (8열)
    tiny_A_1 = tiny_A[:, 8:80]  # s_1에 대응 (72열)
    logger.info("A matrix split into A_0 (%s) and A_1 (%s)", tiny_A_0.shape, tiny_A_1.shape)

    if params.rlwe:
        assert params.N%params.rlwe == 0, f"Total MLWE dimension {params.N} should be a multiple of k={params.rlwe}"
        k = params.rlwe
        n = params.N//k

        params.max_samples //= n
        A_dot_s = np.array([get_mlwe_circ(ak, n, k)@secret for ak in tiny_A]) % params.Q
    else:
        A_dot_s = tiny_A @ secret % params.Q

    # generate the e in the original b=A*s+e
    tiny_e = secret_factory.error(A_dot_s.shape)

    # Will be used by Rb in so that the distribution and dependence of vars are correct
    tiny_b = (A_dot_s + tiny_e) % params.Q

    # full_RA (params.max_samples, N) is the entire set (~4M) of inputs to the transformer
    full_RA = np.zeros((params.max_samples, params.N), dtype=np.int64)

    # full_Rb (NUM_SAMPLES, num_secrets) is a matrix of outputs to the transformer.
    # There are num_secrets columns in Rb; one for each secret.
    # Shape is different for RLWE setting because of the circulant.
    b_shape = (
        (params.max_samples, params.N//params.rlwe, num_secrets)
        if params.rlwe
        else (params.max_samples, num_secrets)
    )
    full_Rb = np.zeros(b_shape, dtype=np.int64)

    # Find the data.prefix file
    data_prefix_path = os.path.join(params.processed_dump_path, "data.prefix")
    if not os.path.isfile(data_prefix_path):
        logger.warning("Copying all data_*.prefix to %s", data_prefix_path)
        # NOTE: you might have to change this glob if you're using threadsafe or not.
        # If you're going after some slurm jobs, then you probably want:
        #
        #    f"{params.processed_dump_path}/*/data_*.prefix"
        #
        # Note the /*/ to match the slurm jobs
        paths = glob.glob(f"{params.processed_dump_path}/*/data_*.prefix")
        with open(data_prefix_path, "w") as outfile:
            for path in tqdm(paths, desc="Copying data.prefix"):
                for line in remove_redundant_rows(path, params.m):
                    outfile.write(line)

    logger.info("Loading data from %s", data_prefix_path)

    n_matrices, n_pairs = 0, 0

    for A, R in read(data_prefix_path, params.m):
        # Remove any rows of R that are all zeros
        # https://stackoverflow.com/a/11188955
        R = R[~np.all(R == 0, axis=1)]

        idxs = list(A.flatten())
        A = tiny_A[idxs]
        assert A.shape == (params.m, params.N)

        b = tiny_b[idxs]

        if params.rlwe:
            assert b.shape == (params.m, params.N//params.rlwe, num_secrets)
        else:
            assert b.shape == (params.m, num_secrets)

        # Compute RA and check if it's too big.
        batch_RA = mod_mult_torch(R, A, params.Q)
        assert batch_RA.dtype == np.int64, (batch_RA.dtype, batch_RA[0][0])
        new_pairs, _ = batch_RA.shape
        end = n_pairs + new_pairs
        if (n_pairs > 0) and (end > params.max_samples):
            logger.info("About to exceed %d. Terminating.", params.max_samples)
            break

        batch_Rb = mod_mult_torch(R, b, params.Q)
        assert batch_Rb.dtype == np.int64, (batch_Rb.dtype, batch_Rb[0][0])
    
        if True: #data_check(batch_RA, batch_Rb, secret, params.rlwe):
            full_RA[n_pairs:end] = batch_RA
            full_Rb[n_pairs:end] = batch_Rb
        else:
            logger.debug("Error too large, ignoring this R")
            continue
        
        # Increment and log if needed.
        n_pairs += new_pairs
        n_matrices += 1
        if n_matrices % params.log_every == 0:
            logger.info(
                "Processed %s matrices from the disk, %s samples.",
                human(n_matrices),
                human(n_pairs),
            )





            
    # Truncate arrays to n_pairs
    full_RA = full_RA[:n_pairs]
    full_Rb = full_Rb[:n_pairs]

    if params.rlwe:
        # Need at least N samples for RLWE in order to have a full circulant.
        test_size = max(test_size//n, params.N + 1)

    # RA를 RA_0 (s_0 대응, 처음 8열)와 RA_1 (s_1 대응, 마지막 72열)로 분리
    # full_RA shape: (n_pairs, N=80)
    full_RA_0 = full_RA[:, :8]  # (n_pairs, 8) - s_0에 대응
    full_RA_1 = full_RA[:, 8:80]  # (n_pairs, 72) - s_1에 대응
    logger.info("full_RA split into RA_0 (%s) and RA_1 (%s)", full_RA_0.shape, full_RA_1.shape)
    
    # full_RA는 더 이상 필요 없으므로 메모리 해제 (RA_0와 RA_1만 유지)
    del full_RA
    gc.collect()

    # RA_1을 80열로 패딩 (앞 8열은 0으로 패딩)
    # full_RA_1: (n_pairs, 72) -> (n_pairs, 80)로 확장
    full_RA_1_padded = np.zeros((full_RA_1.shape[0], params.N), dtype=np.int64)
    full_RA_1_padded[:, 8:80] = full_RA_1
    logger.info("Created padded RA_1: %s", full_RA_1_padded.shape)
    
    # full_RA_1은 더 이상 필요 없으므로 메모리 해제
    del full_RA_1
    gc.collect()

    # Train/Test split (RA_1은 모든 버전에서 동일)
    test_RA_1_padded = full_RA_1_padded[:test_size]
    train_RA_1_padded = full_RA_1_padded[test_size:]
    
    # full_RA_1_padded는 더 이상 필요 없으므로 메모리 해제
    del full_RA_1_padded
    gc.collect()

    def save_and_log(path, arr):
        np.save(path, arr)
        logging.info("Saved %s at %s", os.path.basename(path), path)

    # t의 8가지 경우의 수에 대해 각각 데이터셋 생성
    # t는 s_0와 일치할 때 정답 (s_0는 8행)
    # 각 열(시크릿)에 대해 8개 행 중 어느 위치에 1이 있는지에 따라 8가지 경우
    logger.info("Generating 8 versions of dataset based on t position (s_0, rows 0-7)")

    for t_version in range(8):
        # t_i 버전의 폴더 생성
        version_dump_path = os.path.join(params.dump_path, f"t_{t_version}")
        os.makedirs(version_dump_path, exist_ok=True)

        version_secret_dir = os.path.join(
            version_dump_path,
            f"{params.secret_type}_secrets_h{params.min_hamming}_{params.max_hamming}",
        )
        os.makedirs(version_secret_dir, exist_ok=True)

        logger.info(f"Processing t_{t_version} version...")

        # s를 모두 동일하게 만들었으므로, 모든 시크릿이 동일함
        # s_0의 6번째 비트(인덱스 6, t_6)에만 1이 있으므로, t_version=6만 매칭됨
        # 하지만 각 t_version에 대해 테스트하기 위해, 모든 시크릿을 사용하되
        # t_assumed를 각 t_version에 맞게 설정
        # 실제로는 s_0[6] == 1이므로 t_version=6이 정답이지만,
        # 다른 t_version도 테스트하기 위해 모든 시크릿을 사용
        matching_secrets = np.arange(num_secrets)  # 모든 시크릿 사용
        num_matching = len(matching_secrets)
        logger.info(f"Using all {num_matching} secrets for t_version {t_version} (actual s_0 has 1 at index 6, t_6)")

        # t_version 버전에 대한 t 계산
        # t는 s_0와 같은 형태 (8행)이고, t_version 번째 행만 1, 나머지는 0
        # 원래는 t_assumed 행렬을 만들고 tiny_A_0 @ t_assumed를 계산했지만,
        # t_assumed의 t_version 번째 행만 1이므로 tiny_A_0[:, t_version]만 사용하면 됨

        # 매칭되는 시크릿들의 데이터만 선택
        tiny_b_filtered = tiny_b[:, matching_secrets]  # (num_tiny_samples, num_matching)

        # 최적화: t_assumed는 t_version 번째 행만 1이므로, 직접 계산
        # tiny_A_0 @ t_assumed = tiny_A_0[:, t_version]를 각 열에 복사한 것과 동일
        # 메모리 효율을 위해 in-place 연산 사용
        # modified_tiny_b = tiny_b_filtered - tiny_A_0[:, t_version] (broadcast)
        modified_tiny_b = tiny_b_filtered.copy()  # 복사본 생성 (원본 보존)
        # tiny_A_0[:, t_version]는 (num_tiny_samples,) 형태, 브로드캐스팅으로 각 열에서 빼기
        modified_tiny_b -= tiny_A_0[:, t_version:t_version+1]  # (num_tiny_samples, 1)로 브로드캐스팅
        modified_tiny_b %= params.Q
        logger.info(f"modified_tiny_b shape: {modified_tiny_b.shape}")
        
        # 메모리 해제
        gc.collect()

        # Rb - RA_0 @ t 계산을 위해 full_Rb와 full_RA_0를 사용
        # full_Rb shape: (n_pairs, num_secrets) 또는 (n_pairs, N//rlwe, num_secrets)
        # full_RA_0 shape: (n_pairs, 8)
        # t는 t_version 번째 행만 1, 나머지는 0
        
        # full_Rb에서 매칭되는 시크릿들만 선택
        if params.rlwe:
            full_Rb_filtered = full_Rb[:, :, matching_secrets]  # (n_pairs, N//rlwe, num_matching)
            # RA_0 @ t 계산: full_RA_0는 (n_pairs, 8), t는 (8, num_matching) where t[t_version, :] = 1
            # 하지만 RLWE의 경우 b가 3차원이므로 처리 방법이 다름
            # 일단 RLWE는 나중에 처리하고, 일반 LWE만 먼저 처리
            raise NotImplementedError("RLWE case not yet implemented for Rb - RA_0t")
        else:
            full_Rb_filtered = full_Rb[:, matching_secrets]  # (n_pairs, num_matching)
            # 최적화: t는 t_version 번째 행만 1이므로, full_RA_0[:, t_version]만 사용
            # full_RA_0 @ t = full_RA_0[:, t_version]를 각 열에 복사한 것과 동일
            # 메모리 효율을 위해 in-place 연산 사용
            # modified_full_Rb = full_Rb_filtered - full_RA_0[:, t_version] (broadcast)
            modified_full_Rb = full_Rb_filtered.copy()  # 복사본 생성 (원본 보존)
            # full_RA_0[:, t_version]는 (n_pairs,) 형태, 브로드캐스팅으로 각 열에서 빼기
            modified_full_Rb -= full_RA_0[:, t_version:t_version+1]  # (n_pairs, 1)로 브로드캐스팅
            modified_full_Rb %= params.Q
            logger.info(f"modified_full_Rb shape: {modified_full_Rb.shape}")
            
            # 메모리 해제
            del full_Rb_filtered
            gc.collect()

        # Train/Test split
        test_modified_Rb = modified_full_Rb[:test_size]
        train_modified_Rb = modified_full_Rb[test_size:]

        # params.pkl과 params.json 저장 (각 버전별).  Keep the actual t
        # version in the dataset metadata so Verde can reassemble the fixed
        # prefix during recovery.
        version_params = vars(params).copy()
        version_params["t_version"] = t_version
        with open(os.path.join(version_secret_dir, "params.pkl"), "wb") as fd:
            pickle.dump(version_params, fd)
        with open(os.path.join(version_secret_dir, "params.json"), "w") as fd:
            json.dump(version_params, fd)

        # 파일 저장
        # orig_A, orig_b는 검증용으로 원본 그대로 유지
        # orig_A: 전체 A (80열), orig_b: 전체 b (A @ s + e)
        save_and_log(f"{version_dump_path}/orig_b.npy", tiny_b[:, matching_secrets])  # 원본 b
        save_and_log(f"{version_dump_path}/orig_A.npy", tiny_A)  # 원본 A 전체 (검증용)

        # 학습/테스트용: RA_1 (72열, 80열로 패딩), Rb - RA_0t
        save_and_log(f"{version_dump_path}/test_A.npy", test_RA_1_padded)  # 학습용은 RA_1만 (72열, 80열로 패딩)
        save_and_log(f"{version_dump_path}/train_A.npy", train_RA_1_padded)  # 학습용은 RA_1만 (72열, 80열로 패딩)

        # 학습 목표: s_1 (72행)
        # 학습: RA_1 -> s_1 (72행)
        # 검증: orig_A @ full_s - orig_b = e (full_s는 s_0와 s_1을 합친 것)
        # s_1을 80행으로 패딩 (앞 8행은 0)
        s_1_padded = np.zeros((params.N, num_matching), dtype=np.int64)
        s_1_padded[8:80, :] = s_1[:, matching_secrets]  # s_1 저장 (80행으로 패딩)

        # 각 시크릿별 저장 (매칭되는 시크릿들만)
        save_and_log(f"{version_secret_dir}/secret.npy", s_1_padded)  # s_1 저장 (80행으로 패딩)

        # matching_secrets 인덱스를 (h, seed_i) 쌍으로 변환
        # 메모리 효율을 위해 한 번에 하나씩 저장
        for idx, secret_i in enumerate(tqdm(matching_secrets, desc=f"Saving t_{t_version} secrets")):
            # secret_i를 (h, seed_i)로 역변환
            h = params.min_hamming + secret_i // params.num_secret_seeds
            seed_i_local = secret_i % params.num_secret_seeds

            np.save(f"{version_secret_dir}/secret_{h}_{seed_i_local}.npy", s_1_padded[..., idx])
            np.save(f"{version_secret_dir}/orig_b_{h}_{seed_i_local}.npy", tiny_b[:, secret_i])
            np.save(f"{version_secret_dir}/test_b_{h}_{seed_i_local}.npy", test_modified_Rb[..., idx])
            np.save(f"{version_secret_dir}/train_b_{h}_{seed_i_local}.npy", train_modified_Rb[..., idx])
            np.save(f"{version_secret_dir}/reduced_b_{h}_{seed_i_local}.npy", modified_full_Rb[..., idx])

        # 이 버전 처리 완료 후 메모리 해제
        del modified_full_Rb, test_modified_Rb, train_modified_Rb
        del tiny_b_filtered, modified_tiny_b
        del s_1_padded
        gc.collect()
        
        logger.info(f"Completed t_{t_version} version: saved to {version_dump_path}")

    logger.info(
        "Saved all 8 versions (t_0 to t_7) of datasets at %s",
        params.dump_path,
    )


# Plotting
def plot(reduced_A, orig_A):
    """Plots A's column-wise standard deviation."""
    import matplotlib.pyplot as plt

    x = np.arange(params.N)
    fig, ax = plt.subplots()

    reduced_std = centered(reduced_A, params.Q).std()

    y = centered(reduced_A, params.Q).std(0)
    ax.plot(x, y, label="Reduced")

    # Original data
    orig_std = centered(orig_A, params.Q).std()

    y = centered(orig_A, params.Q).std(0)
    ax.plot(x, y, label="Original")

    ax.set_ylim(ymin=0)
    ax.set_xlabel("A[i]")
    ax.set_ylabel("Std. Dev.")
    ax.set_title(
        f"{os.path.basename(params.processed_dump_path)} (Reduction {human(reduced_std)}/{human(orig_std)} = {reduced_std/orig_std:.2f})"
    )
    ax.legend()

    fig.savefig(
        f"{params.dump_path}/R_A_{params.N}_{params.logq}_omega{params.omega}.pdf"
    )
    fig.savefig(
        f"{params.dump_path}/R_A_{params.N}_{params.logq}_omega{params.omega}.png"
    )

def generate_usvp_secret():
    if params.secret_type == "binary":
        secret_factory = BinaryFactory(sigma=params.sigma)
    elif params.secret_type == "ternary":
        secret_factory = TernaryFactory(sigma=params.sigma)
    elif params.secret_type == "gaussian":
        secret_factory = GaussianFactory(sigma=params.sigma)
    elif params.secret_type == "binomial":
        secret_factory = BinomialFactory(gamma=params.gamma)
    else:
        raise ValueError(params.secret_type)

    logger.info("Creating %s secret", secret_factory.type_)
    secret = secret_factory.new(
        params.N, params.min_hamming, params.max_hamming, params.num_secret_seeds
    )
    def save_and_log(path, arr):
        np.save(path, arr)
        logging.info("Saved %s at %s", os.path.basename(path), path)

    # Since we dump the secrets down a directory.
    save_and_log(f"{params.secret_dir}/secret.npy", secret)


def describe(reduced_A, orig_A):
    """
    Calculates and reports:
    * The number of samples in reduced_A
    * the std deviation
    * the reduction to 4 decimal places
    * the number of cruel bits
    * recommended hamming weights for secrets that will take 100B or fewer guesses.
    * the block sizes used during BKZ
    * the variance and std of the error distribution after reduction.
    """
    # Number of samples
    logger.info("Number of samples: %d", reduced_A.shape[0])

    # Std deviation
    logger.info("Regular std:  %.4f", reduced_A.std())
    logger.info("Centered std: %.4f", centered(reduced_A, params.Q).std())

    # Reduction
    logger.info("Regular reduction:  %.4f", reduced_A.std() / orig_A.std())
    logger.info(
        "Centered reduction: %.4f", centered(reduced_A, params.Q).std() / centered(orig_A, params.Q).std()
    )

    # Cruel bits
    orig_std = centered(orig_A, params.Q).std()
    # bits above this threshold are cruel.
    cruel_threshold = 0.5 * orig_std
    bitwise_std = centered(reduced_A, params.Q).std(axis=0)
    n_cruel_bits = (bitwise_std > cruel_threshold).sum()
    logger.info("Cruel bits:  %d", n_cruel_bits)

    # Recommended hamming weights
    max_brute_force_bits = 0
    while math.comb(n_cruel_bits, max_brute_force_bits + 1) < 100_000_000_000:
        max_brute_force_bits += 1

    max_h = int(max_brute_force_bits / n_cruel_bits * params.N)

    logger.info(
        "Recommended max hamming: %d, max brute force bits: %d , choices: %s",
        max_h,
        max_brute_force_bits,
        human(math.comb(n_cruel_bits, max_brute_force_bits)),
    )

    # BKZ block sizes
    # logger.info(
    #     "BKZ block sizes: %d, %d", params.bkz_block_size1, params.bkz_block_size2
    # )

    # Std dev of (RA @ s - RB) % Q
    # Do this for 10 secrets. Report all of the values, the mean and the std.
    diff_stds = []
    try:
        diff_files = [file for file in os.listdir(params.secret_dir) if "diff" in file]
        for diff_file in shuffled(diff_files, rng)[:50]:
            diff = np.load(os.path.join(params.secret_dir, diff_file))
            diff_stds.append((diff % params.Q).std() / params.Q)

        if diff_stds:
            logger.info(
                "Std deviation of (RA*s - Rb mod Q) / Q: %.3f (%.3f)",
                np.mean(diff_stds),
                np.std(diff_stds),
            )
    except Exception as e:
        logger.warning("Error calculating diff std: %s", e)


def get_loaded_params():
    # Copy the params file up from an experiment subdirectory, if it doesn't exist in
    # reload_data.
    params_pkl_path = os.path.join(params.processed_dump_path, "params.pkl")
    if not os.path.exists(params_pkl_path):
        for dirpath, dirnames, filenames in os.walk(params.processed_dump_path):
            if "params.pkl" not in filenames:
                continue
            src = os.path.join(dirpath, "params.pkl")
            dst = os.path.join(params.processed_dump_path, "params.pkl")
            try:
                shutil.copy(src, dst)
                params_pkl_path = dst
            except PermissionError:
                params_pkl_path = src

    # Grab parameters from the processed data
    with open(params_pkl_path, "rb") as fd:
        loaded_params = pickle.load(fd)
    if not isinstance(loaded_params, dict):
        loaded_params = vars(loaded_params)

    return loaded_params


def resolve_orig_A_path(raw_path):
    """Resolve an original-A path independently of the process cwd."""
    raw_path = os.path.expanduser(str(raw_path))
    candidates = []
    if os.path.isabs(raw_path):
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                os.path.abspath(raw_path),
                os.path.abspath(os.path.join(params.processed_dump_path, raw_path)),
                os.path.abspath(
                    os.path.join(os.path.dirname(params.processed_dump_path), raw_path)
                ),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", raw_path)),
            ]
        )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not resolve the original A matrix path. "
        f"Configured path={raw_path!r}; checked:\n" + "\n".join(candidates)
    )


if __name__ == "__main__":
    params = get_params()
    if params.seed < 0:
        params.seed = int(time.time())
    os.makedirs(params.dump_path, exist_ok=True)
    logger = create_this_logger(params)
    rng = init_rng(params.seed, logger)

    # If a secret path is provided, make sure params are updated accordingly. 
    if os.path.isfile(params.secret_path):
        # Infer characteristics from dirname (min_hamming, max_hamming, secret_type) 
        dirname = os.path.dirname(params.secret_path).split("/")[-1]
        try:
            s = np.load(params.secret_path)
            params.secret_type = dirname.split('_')[0]
            params.min_hamming = int(dirname.split('_')[-2][1:])
            params.max_hamming = int(dirname.split('_')[-1])
            params.num_secret_seeds = s.shape[-1] // (params.max_hamming - params.min_hamming + 1)
            print(params.min_hamming, params.max_hamming)
        except Exception as e:
            print(f"Failed with error {e}")
            assert False == True, 'Could not infer secret characteristics from dirname'

     # If you're running only_secrets, you only want secret.npy file. 
    if "only_secrets" in params.actions: 
        # Just make the secret file and exit.
        params.dump_path = os.path.dirname(params.dump_path) # We have custom experiment ID for only secrets.

        # Make a path to dump the secrets
        params.secret_dir = os.path.join(params.dump_path,
        f"secret_N{params.N}_{params.secret_type}_{params.min_hamming}_{params.max_hamming}",
        )
        os.makedirs(params.secret_dir, exist_ok=True)

        # Make the secret
        generate_usvp_secret()
        sys.exit(0) # Then you're done


    loaded_params = get_loaded_params()

    params.N, params.Q = loaded_params["N"], loaded_params["Q"]
    params.logq = int(np.ceil(np.log2(params.Q)))
    if "lll_penalty" in loaded_params:
        params.omega = loaded_params["lll_penalty"]
    else:
        params.omega = loaded_params["omega"]
    if "bkz_block_size" in loaded_params:
        params.bkz_block_size1 = loaded_params["bkz_block_size"]
    else:
        params.bkz_block_size1 = loaded_params["bkz_block_size1"]
    #params.bkz_block_size2 = loaded_params["bkz_block_size2"]

    # std. dev. threshold for when matrices were written to data.prefix.
    if "write_threshold" in loaded_params:
        params.write_threshold = loaded_params["write_threshold"]
    else:
        params.write_threshold = float("inf")
        for threshold_key in ("threshold1", "threshold2", "threshold"):
            if threshold_key in loaded_params:
                params.write_threshold = loaded_params[threshold_key]
                break

    dirname = f"A_b_{params.N}_{params.logq}_omega{params.omega}"
    if params.tag:
        dirname += f"_{params.tag}"

    params.secret_dir = os.path.join(
        params.dump_path,
        f"{params.secret_type}_secrets_h{params.min_hamming}_{params.max_hamming}",
    )
    if "secrets" in params.actions:
        os.makedirs(params.secret_dir, exist_ok=True)

    if "m" not in loaded_params or loaded_params["m"] == -1:
        params.m = loaded_params["N"]
    else:
        params.m = loaded_params["m"]

    configured_orig_A = params.orig_A_path
    if not configured_orig_A:
        if "reload_data" in loaded_params:
            configured_orig_A = loaded_params["reload_data"]
        else:
            configured_orig_A = loaded_params["orig_A_path"]
    params.orig_A_path = resolve_orig_A_path(configured_orig_A)


    if "secrets" in params.actions:
        # params.pkl과 params.json은 각 t_i 버전 폴더에 저장됨
        generate()

    # plot과 describe는 각 t_i 버전별로 실행
    if "plot" in params.actions or "describe" in params.actions:
        for t_version in range(8):
            version_dump_path = os.path.join(params.dump_path, f"t_{t_version}")

            # 각 버전의 파일 체크
            train_A_path = f"{version_dump_path}/train_A.npy"
            orig_A_path = f"{version_dump_path}/orig_A.npy"

            if not os.path.exists(train_A_path) or not os.path.exists(orig_A_path):
                logger.warning(f"Skipping t_{t_version}: files not found")
                continue

            reduced_A = np.load(train_A_path)
            orig_A = np.load(orig_A_path)

            logger.info(f"Processing plot/describe for t_{t_version}...")

            # 각 버전의 secret_dir 업데이트
            version_secret_dir = os.path.join(
                version_dump_path,
                f"{params.secret_type}_secrets_h{params.min_hamming}_{params.max_hamming}",
            )
            old_secret_dir = params.secret_dir
            params.secret_dir = version_secret_dir

            if "plot" in params.actions:
                # plot 함수 내부에서 params.dump_path를 사용하므로 임시 변경
                old_dump_path = params.dump_path
                params.dump_path = version_dump_path
                plot(reduced_A, orig_A)
                params.dump_path = old_dump_path

            if "describe" in params.actions:
                describe(reduced_A, orig_A)

            # 원래 secret_dir로 복원
            params.secret_dir = old_secret_dir

        logger.info("Completed plot/describe for all 8 versions")
