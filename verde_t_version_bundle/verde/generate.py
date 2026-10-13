# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import random
import argparse
import joblib
import numpy as np
import torch
import os
import logging

from src import utils
from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import bool_flag, initialize_exp, create_logger
from src.generate.export import Generator
from src.generate.genSamples import BKZReducedRLWE, RA_Rb, BenchmarkBKZ
from multiprocessing import Process, Manager
from joblib import Parallel, delayed

np.seterr(all='raise')


def get_parser():
    """
    Generate a parameters parser.
    """
    # generate.py는 단계별로 필요한 설정이 꽤 많아서
    # 커맨드라인 인자를 여기서 한 번에 정리한다.
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="",
                        help="Experiment dump path")
    parser.add_argument("--resume_path", type=str, default="",
                        help="Path to load the checkpoints")
    parser.add_argument("--secret_dir", type=str, default="",
                        help="Path to the secret directory, for step Ab")
    parser.add_argument("--exp_name", type=str, default="debug",
                        help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")
    parser.add_argument("--step", type=str, default="RA",
                        help="data generation step, RA, RA_tiny1, RA_tiny2, BKZ, or Ab")

    # iteration
    parser.add_argument("--env_base_seed", type=int, default=-1,
                        help="Base seed for environments (-1 to use timestamp seed)")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Number of sentences per batch")
    parser.add_argument("--epoch_size", type=int, default=4000,
                        help="number of matrices to generate for each worker")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of CPU workers for DataLoader")

    # Load data
    parser.add_argument("--reload_data", type=str, default="",
                        help="Directory to load the dataset from the disk. For RA_tiny, it's A; for Ab, it's R,A")
    parser.add_argument("--reload_perm", type=str, default="",
                        help="Directory to tinyA with permuted cols")
    parser.add_argument("--reload_size", type=int, default=100000,
                        help="Reloaded number of matrices")

    # debug
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug multi-GPU / multi-node within a SLURM job")
    parser.add_argument("--debug", default=False, action="store_true", 
                        help="Enable all debug flags")

    # CPU / multi-gpu / multi-node
    parser.add_argument("--cpu", type=bool_flag, default=True,
                        help="Run on CPU")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")

    parser.add_argument("--timeout", type=int, default=10,
                        help="Seconds before timeout in generate")
    parser.add_argument("--max_timeout_count", type=int, default=10,
                        help="Maximum Number of timeout in a row before stopping generation")
    
    # LWE
    parser.add_argument("--lwe", type=bool_flag, default=False, 
                        help="LWE if true. RLWE by default. Ignored by tiny samples, which loads from reload data")
    parser.add_argument("--N", type=int, default=-1, 
                        help="dimension of matrix")
    parser.add_argument("--Q", type=int, default=-1, 
                        help="modulo")
    parser.add_argument("--num_secret_seeds", type=int, default=5,
                        help="how many seeds for secrets")
    parser.add_argument("--min_hamming", type=int, default=3,
                        help="min hamming weight when generating secrets")
    parser.add_argument("--max_hamming", type=int, default=20,
                        help="max hamming weight when generating secrets. Used in step2_Ab and benchmarking")
    parser.add_argument("--secret_type", type=str, default="binary",
                        help="binary, ternary, gaussian, or binomial")
    parser.add_argument("--sigma", type=float, default=3, 
                        help='sigma for gaussian error')
    parser.add_argument("--gamma", type=int, default=2, 
                        help='gamma for binomial error')
    parser.add_argument("--correctQ", type=bool_flag, default=False, 
                        help='flip the Q range to be within -Q/2 and Q/2?')

    # LLL/BKZ reduction parameters
    parser.add_argument("--float_type", type=str, default="double", 
                        help="double, long double, dpe, dd, qd, or mpfr_<precision>")
    parser.add_argument("--lll_penalty", type=int, default=10, 
                        help="penalty on norm of LLL Reduced A")
    parser.add_argument("--algo", type=str, default='BKZ2.0', 
                        help='Phase 1, algorithm to use: BKZ or BKZ2.0')
    parser.add_argument("--lll_delta", type=float, default=0.96, 
                        help="Phase 1, hermite factor for LLL")
    parser.add_argument("--bkz_block_size", type=int, default=30, 
                        help="Phase 1, block size of the BKZ reduction")
    parser.add_argument("--algo2", type=str, default='BKZ2.0', 
                        help='Phase 2, algorithm to use: BKZ or BKZ2.0')
    parser.add_argument("--lll_delta2", type=float, default=0.99, 
                        help="Phase 2, hermite factor for LLL")
    parser.add_argument("--bkz_block_size2", type=int, default=40, 
                        help="Phase 2, block size of the BKZ reduction")
    parser.add_argument("--m", type=int, default=-1,
                        help="number of samples used in BKZ reduction, defaults to N")
    parser.add_argument("--threshold", type=float, default=-1,
                        help="the threshold to terminate reduction")
    parser.add_argument("--threshold2", type=float, default=0.5,
                        help="the threshold to go to phase 2")
    parser.add_argument("--rnorm", type=float, default=1.0, 
                        help='threshold for ||R||/q in step Ab')
    parser.add_argument("--use_polish", type=bool_flag, default=False, 
                        help='whether to polish after bkz')

    return parser
    
def initialize(params, logger, thread):

    # step에 따라 실제 샘플 생성기가 완전히 달라진다.
    if params.step in ["RA", "RA_tiny1", "RA_tiny2"]:

        # RA 계열은 BKZ로 줄인 RLWE 샘플을 만든다.
        sampleGen = BKZReducedRLWE(params, thread)

    elif params.step == "BKZ":

        # BKZ는 진행 상황을 누적 저장하므로
        # 첫 실행에서 결과 컨테이너를 먼저 만들어 둔다.
        if not os.path.isfile(os.path.join(params.dump_path, 'results.pkl')):

            keys = []

            # 여기서는 Hamming weight마다 5개 실험을 기록한다.
            for expNum in range(5):

                # max_hamming
                keys += [(expNum, params.max_hamming)]

                # max_hamming - 1
                keys += [(expNum, params.max_hamming - 1)]

            # key는 (실험 번호, hamming weight),
            # value는 시간/성공 여부가 쌓이는 리스트다.
            pickle.dump(
                dict([(key, []) for key in keys]),
                open(os.path.join(params.dump_path, 'results.pkl'), 'wb')
            )

        # BKZ benchmark용 생성기로 분기한다.
        sampleGen = BenchmarkBKZ(params, thread)

    elif params.step == "Ab":

        # Ab 단계는 이미 저장된 prefix 파일을 다시 읽어서 사용한다.
        sampleGen = RA_Rb(params)

    else:

        # 알 수 없는 step은 여기서 바로 종료한다.
        logger.info(
            'Step not recognized. Must be one of RA, RA_tiny1, RA_tiny2, Ab, or BKZ'
        )
        exit()

    # step별 생성기와 export 래퍼를 연결한다.
    generator = Generator(params, sampleGen)

    # 실제 반복, 저장, 통계 출력은 Generator가 담당한다.
    return generator

def get_data_one_worker(i, params):

    # 각 worker가 같은 로그 파일에 기록하지만
    # 형식은 통일해서 뒤에서 읽기 쉽게 둔다.
    logger = create_logger(os.path.join(params.dump_path, 'train.log'), rank=0)

    # worker 전용 샘플러를 초기화한다.
    gen = initialize(params, logger, i)

    # worker가 만든 equation 수를 세기 시작한다.
    gen.n_equations = 0

    # 목표 개수(epoch_size)에 도달할 때까지 생성한다.
    while gen.n_equations < gen.epoch_size:

        # None이면 더 이상 읽거나 만들 샘플이 없다는 뜻이다.
        if gen.export_data() is None:
            logger.info("============ End of file ============")

            # step별 마무리 저장을 수행한다.
            gen.end_epoch()
            return

        # 샘플 생성과 저장을 한 번 수행한다.
        gen.iter()
    
def main(params):

    # SLURM / 멀티 GPU 환경 변수를 먼저 정리한다.
    init_distributed_mode(params)

    # params를 기록하고 logger를 세팅한다.
    logger = initialize_exp(params)

    # SLURM에서 강제 종료 신호가 오면 재큐잉/정리를 하도록 연결한다.
    if params.is_slurm_job:
        init_signal_handler()

    # CPU 모드와 분산 GPU 모드는 같이 쓰지 않는다.
    if params.cpu:
        assert not params.multi_gpu
    else:
        # GPU로 돌릴 때는 CUDA가 실제로 활성화돼 있어야 한다.
        assert torch.cuda.is_available()

    # 유틸 함수에서 볼 전역 CUDA 스위치를 맞춘다.
    utils.CUDA = not params.cpu

    # seed가 없으면 실행마다 다른 난수를 사용한다.
    if params.env_base_seed < 0:
        params.env_base_seed = np.random.randint(1_000_000_000)

    # 이 머신에서 실제로 사용할 수 있는 CPU 개수를 본다.
    n_cpu = joblib.cpu_count()

    # CPU보다 많은 worker는 만들지 않는다.
    n_jobs = min(n_cpu, params.num_workers)

    logger.info(
        f" Nb CPU: {n_cpu} and Nb worker: {params.num_workers}"
    )

    # worker들을 병렬로 돌려서 데이터를 생성한다.
    Parallel(
        n_jobs=n_jobs,
        prefer='threads'
    )(
        delayed(get_data_one_worker)(n, params)
        for n in range(n_jobs)
    )

if __name__ == '__main__':

    # 커맨드라인 인자를 읽을 parser를 만든다.
    parser = get_parser()

    # argparse 결과를 params 객체로 받는다.
    params = parser.parse_args()

    # debug 모드는 결과를 찾기 쉽도록 이름과 ID를 고정한다.
    if params.debug:

        # 결과 디렉터리가 항상 debug 아래에 생성되도록 맞춘다.
        params.exp_name = 'debug'

        # exp_id가 없으면 충돌을 피하려고 임의 ID를 붙인다.
        if params.exp_id == '':
            params.exp_id = 'debug_%08i' % random.randint(0, 100000000)

        # 로컬 디버깅 시 SLURM 분기를 타지 않게 만든다.
        params.debug_slurm = True

    # Ab/BKZ는 이전 단계에서 만든 설정을 그대로 이어받아야 한다.
    if params.step in ["Ab", "BKZ"]:

        # 이전 실험의 params.pkl을 읽어서 현재 설정을 보정한다.
        loaded_params = pickle.load(
            open(os.path.join(params.reload_data, 'params.pkl'), 'rb')
        )

        # 저장 형태가 Namespace면 dict처럼 쓰기 위해 바꾼다.
        if type(loaded_params) != dict:
            loaded_params = loaded_params.__dict__

        # 차원 N과 모듈로 Q는 원본 데이터와 맞춰야 한다.
        params.N = loaded_params['N']
        params.Q = loaded_params['Q']

        if params.step == "BKZ":

            # BKZ는 secret_type도 원본과 맞춰야 비교가 된다.
            params.secret_type = (
                loaded_params['secret_type']
                if 'secret_type' in loaded_params
                else 'binary'
            )

        else:

            # m 값이 없으면 N을 기본값으로 쓴다.
            if 'm' not in loaded_params or loaded_params['m'] == -1:
                params.m = loaded_params['N']
            else:
                params.m = loaded_params['m']

            # tiny 데이터셋 여부를 기록해 두고 아래에서 분기한다.
            params.tiny1 = loaded_params['step'] == 'RA_tiny1'
            params.tiny2 = loaded_params['step'] == 'RA_tiny2'

            # tiny 샘플 복원에 쓸 원본 A 경로를 보관한다.
            params.orig_A_path = loaded_params['reload_data']

    # 준비가 끝났으니 실제 생성 루프로 들어간다.
    main(params)
