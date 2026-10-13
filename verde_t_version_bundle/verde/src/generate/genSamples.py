# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import pickle
import numpy as np
from time import time
from scipy.linalg import circulant
from logging import getLogger
from fpylll import FPLLL, LLL, BKZ, GSO, IntegerMatrix
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from ..utils import TimeoutError, timeout
import multiprocessing

logger = getLogger()
FLOAT_UPGRADE = {
    'double': 'dd',
    'long double': 'dd',
    'dd': 'qd',
    'qd': 'mpfr_250'
}
MAX_TIME = 60

class BKZReducedRLWE():
    def __init__(self, params, thread):
        # RA 단계에서는 RLWE 샘플을 만들고 BKZ로 줄이는 과정을 반복한다.
        self.params = params
        self.N = params.N
        self.Q = params.Q
        self.m = params.N if params.m == -1 else params.m
        self.longtype = np.log2(params.Q) > 30
        self.set_float_type(params.float_type)
        self.counter = 0
        self.timeout = params.timeout
        
        self.step = params.step
        assert (params.reload_data == "") == (self.step == "RA")
        # tiny step은 미리 저장된 A를 다시 읽어와서 쓴다.
        if self.step != "RA":
            if params.reload_perm == "":
                self.tiny_A = np.load(params.reload_data)
                logger.info(f'Generating R,A with tiny samples at {params.reload_data}.')
            else:
                self.tiny_A = np.load(params.reload_perm)
                logger.info(f'Generating R,A with tiny samples at {params.reload_perm}.')
            assert self.tiny_A.shape[1] == self.N

        self.matrix_filename = os.path.join(params.dump_path, f"matrix_{thread}.npy")
        self.resume_filename = os.path.join(params.resume_path, f"matrix_{thread}.npy")
        # 이전 실행이 남긴 matrix가 있으면 이어서 쓴다.
        if os.path.isfile(self.resume_filename):
            mat_to_save = np.load(self.resume_filename)
            np.save(self.matrix_filename, mat_to_save)
        self.seed = [params.global_rank, params.env_base_seed, thread]
        # env_base_seed: different jobs will generate different data
        # thread: different workers will not generate same data
        logger.info(f"Random generator seed: {self.seed}. Resuming from {self.resume_filename}. ")

    def set_float_type(self, float_type):
        # fplll은 float precision에 민감해서, 필요하면 정밀도를 직접 올린다.
        self.float_type = float_type
        parsed_float_type = float_type.split('_')
        if len(parsed_float_type) == 2:
            self.float_type, precision = parsed_float_type
            assert self.float_type == 'mpfr'
            FPLLL.set_precision(int(precision))

    def timedout_bkz(self, return_dict):
        # 별도 프로세스에서 BKZ를 돌려서 timeout 제어를 단순하게 만든다.
        if os.path.isfile(self.matrix_filename):
            A_Ap = np.load(self.matrix_filename)
            UT_or_AT, Ap = A_Ap[:, :self.m], A_Ap[:, self.m:]
        else:
            A, Ap = self.get_A_Ap()
            UT_or_AT = A.T # To have num_cols = m, U and A are always transposed

        self.algo, self.upgraded = self.params.algo, False
        self.block_size, self.delta = self.params.bkz_block_size, self.params.lll_delta
        if self.calc_std(Ap) < self.params.threshold2: # use Phase 2 params
            self.algo, self.upgraded = self.params.algo2, True
            self.block_size, self.delta = self.params.bkz_block_size2, self.params.lll_delta2

        # reduction이 멈추거나 파라미터를 바꿔야 할 때까지 반복한다.
        param_change = True
        while param_change:
            Ap, param_change = self.run_bkz(return_dict, UT_or_AT, Ap)

        # 다음 반복을 위해 체크포인트를 최신 상태로 다시 저장한다.
        newA, newAp = self.get_A_Ap()
        self.save_mat(newA.T, newAp)

        R = (Ap[:,:self.m] / self.params.lll_penalty).astype(int)
        return_dict["R"] = R
        return_dict["XT"] = UT_or_AT[:len(newA.T)]

    def run_bkz(self, return_dict, UT_or_AT, Ap):
        # fplll이 기대하는 IntegerMatrix 형태로 바꾼 뒤 BKZ를 실행한다.
        fplll_Ap = IntegerMatrix.from_matrix(Ap.tolist())
        M = GSO.Mat(fplll_Ap, float_type=self.float_type, update=True)
        bkz_params = BKZ.Param(self.block_size, delta=self.delta, max_time=MAX_TIME)
        if self.algo == 'BKZ':
            L = LLL.Reduction(M, delta=self.delta)
            BKZ_Obj = BKZ.Reduction(M, L, bkz_params)
        else:
            BKZ_Obj = BKZ2(M)

        while True:
            try: 
                BKZ_Obj() if self.algo == 'BKZ' else BKZ_Obj(bkz_params)
            except: 
                # BKZ가 수치 정밀도 문제를 만나면, 더 높은 precision으로 다시 시도한다.
                self.set_float_type(FLOAT_UPGRADE[self.float_type])
                logger.info(f'Error running bkz. Upgrading to float type {self.float_type}.')
                return Ap, True
            
            Ap = np.zeros((self.m+self.N, self.m+self.N), dtype=int)
            fplll_Ap.to_matrix(Ap)
            oldstddev, polishtime = self.calc_std(Ap), time()
            newstddev = self.polish(Ap) if self.params.use_polish else oldstddev
            # 표준편차가 충분히 낮아지면 여기서 바로 멈춘다.
            if newstddev < self.params.threshold:
                logger.info(f'stddev = {newstddev}. Exporting {self.matrix_filename}')
                return Ap, False

            self.save_mat(UT_or_AT, Ap)
            logger.info(f'stddev = {newstddev}. Saved progress at {self.matrix_filename}')
            if not self.upgraded and newstddev < self.params.threshold2:
                # 1차 목표에 도달하면 더 강한 설정의 2단계로 넘어간다.
                self.algo, self.upgraded = self.params.algo2, True
                self.block_size, self.delta = self.params.bkz_block_size2, self.params.lll_delta2
                logger.info(f'Upgrading to delta = {self.delta} and block size = {self.block_size}.')
                return Ap, True
            if oldstddev - newstddev > 0:
                logger.info(f'stddev reduction: {oldstddev - newstddev}, time {time() - polishtime}. ')
                return Ap, True
        
    def save_mat(self, X, Y):
        # 현재까지의 상단 블록 X와 하단 블록 Y를 하나의 checkpoint로 합친다.
        mat_to_save = np.zeros((len(Y), len(Y)+self.m)).astype(int)
        mat_to_save[:len(X), :self.m] = X
        mat_to_save[:, self.m:] = Y
        np.save(self.matrix_filename, mat_to_save)

    def calc_std(self, X):
        # 모듈로 Q 기준으로 실제 분포가 얼마나 퍼져 있는지 본다.
        mat = X[:, self.m:] % self.Q # mat is now a new matrix with entries copied from X
        mat[mat > self.Q//2] -= self.Q
        return np.sqrt(12) * np.std(mat[np.any(mat!=0, axis=1)]) / self.Q

    def polish(self, X):
        # BKZ 이후에 간단한 local search로 표준편차를 더 줄일 수 있다.
        if self.longtype:
            X = X.astype(np.longdouble)
        g, old = np.inner(X, X), np.inf # Initialize the Gram matrix
        while np.std(X) < old:
            old = np.std(X)
            # 현재 Gram 행렬을 바탕으로 각 벡터의 투영 계수를 계산한다.
            c = np.round(g / np.diag(g)).T.astype(int)
            c[np.diag_indices(len(X))] = 0
            c[np.diag(g)==0] = 0
            
            sq = np.diag(g) + c*((c.T*np.diag(g)).T - 2*g)  # 각 벡터를 기준으로 한 norm 후보
            s = np.sum(sq, axis=1)  # 가장 좋아 보이는 축을 고른다.
            it = np.argmin(s)
            X -= np.outer(c[it], X[it])  # 선택한 벡터를 기준으로 나머지를 정리한다.
            g += np.outer(c[it], g[it][it] * c[it] - g[it]) - np.outer(g[:,it], c[it])
        return self.calc_std(X)

    def generate(self):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        p = multiprocessing.Process(target=self.timedout_bkz, name="data generation", args=(return_dict,))
        p.start()
        start = time()
        curr = time() - start
        while curr < self.timeout:
            curr = time() - start
            if not p.is_alive():
                break
        p.kill()
        p.join()
        if curr >= self.timeout:
            raise TimeoutError
        try:
            R = return_dict["R"]
            XT = return_dict["XT"]
        except KeyError:
            raise TimeoutError
        self.counter += 1
        # 너무 자주 로깅하지 않도록 일정 주기로만 상태를 남긴다.
        if self.counter % 200 == 0:
            logger.info(f"Counter {self.counter}; mean norm of rows of R = {np.linalg.norm(R, axis = 1).mean()} ")
        return XT.T, R.T # XT.T would be A or U (tiny samples). shapes: A: (m, N), U: (m, 1 or 4), RT: (m, m+N)
    
    def get_A_Ap(self):
        # 현재 step에 맞는 A와 BKZ 입력용 embedding matrix를 만든다.
        rng = np.random.RandomState(self.seed + [int(time())])
        if self.step == "RA_tiny1": # Deprecated, and not supporting m!=N
            combo_size = 4
            U = np.zeros((self.N, combo_size)).astype(int)
            A = np.zeros((self.N, self.N)).astype(int)
            for j in range(self.N):
                U[j] = rng.choice(len(self.tiny_A), size = combo_size, replace=False)
                for k in U[j]:
                    A[j] += self.tiny_A[k]
        elif self.step == "RA_tiny2":
            idxs = rng.choice(len(self.tiny_A), size = self.m, replace=False)
            U = idxs.reshape((self.m, 1))
            A = self.tiny_A[idxs]
        else: # regular
            if self.params.lwe:
                A = rng.randint(0, self.Q, size=(self.m, self.N), dtype=np.int64)
            else:
                a = rng.randint(0, self.Q, size=self.N, dtype=np.int64)
                A = circulant(a)
                tri = np.triu_indices(self.N, 1)
                A[tri] *= -1
        A = A % self.Q
        assert (np.min(A) >= 0) and (np.max(A) < self.Q)
        
        # BKZ가 다루기 쉬운 형태로 만든다.
        Ap = np.zeros((self.m+self.N, self.m+self.N), dtype = int)
        Ap[self.N:, :self.m] = np.identity(self.m, dtype = int)*self.params.lll_penalty
        Ap[self.N:, self.m:] = A
        Ap[:self.N, self.m:] = np.identity(self.N, dtype = int)*self.Q

        if self.step == "RA":
            return A, Ap # A.shape = mxN, Ap.shape = (m+N)*(m+N)
        else:
            return U, Ap # U.shape = mx1, Ap.shape = (m+N)*(m+N)

class BenchmarkBKZ():
    def __init__(self, params, thread):
        # BKZ benchmark는 미리 만든 secret 세트를 기준으로 성공 여부를 측정한다.
        self.N = params.N
        self.Q = params.Q
        self.block_size = params.bkz_block_size
        self.sigma = params.sigma
        self.max_hamming = params.max_hamming
        self.init_bkz_params(params.secret_type)
        self.set_float_type(params.float_type)

        self.expNum, self.hamming = thread // 2, self.max_hamming - thread % 2
        # secret.npy에는 여러 해밍 가중치를 가진 비밀 벡터들이 저장돼 있다.
        secrets = np.load(os.path.join(params.reload_data, 'secret.npy'))
        cols = np.where(np.sum(secrets != 0, axis=0) == self.hamming)[0]
        assert len(cols) > self.expNum
        self.s = (secrets[:, cols[self.expNum]]).reshape((self.N, 1))
        assert sum(self.s != 0) == self.hamming

        self.results_path = os.path.join(params.dump_path, 'results.pkl')
        self.matrix_filename = os.path.join(params.dump_path, f"matrix_{thread}.npy")
        self.seed = [params.global_rank, params.env_base_seed, thread]
        # env_base_seed: different jobs will generate different data
        # thread: different workers will not generate same data
        logger.info(f"data folder = {self.expNum}, h = {self.hamming}, random generator seed: {self.seed}")
        
    def init_bkz_params(self, secret_type):
        # secret 타입에 따라 BKZ embedding에서 사용할 weight와 delta를 정한다.
        n, sigma = self.N, self.sigma
        q = 2 ** np.ceil(np.log2(self.Q))
        w = 1 # weight for gaussian secret
        if secret_type == 'binary':
            w = np.round(np.sqrt(2) * self.sigma)
        elif secret_type == 'ternary':
            w = np.round(np.sqrt(1.5) * self.sigma)
        d = 2*n*np.log(q/w)  /  np.log( q / ( np.sqrt(2*np.pi*np.e) * sigma) )
        logdelta = np.log(q / (np.sqrt(2*np.pi*np.e) * sigma)) ** 2  /  (4*n*np.log(q/w))
        self.weight = int(w)
        self.m = int(np.ceil(d)) - n - 1
        self.delta = min(np.round(np.e**logdelta, 4), 1)
        logger.info(f"w = {self.weight}, delta = {self.delta}, d = {self.m+self.N+1}")

    def set_float_type(self, float_type):
        self.float_type = float_type
        parsed_float_type = float_type.split('_')
        if len(parsed_float_type) == 2:
            self.float_type, precision = parsed_float_type
            assert self.float_type == 'mpfr'
            FPLLL.set_precision(int(precision))

    def save_mat(self, X, Y):
        # 진행 중인 embedding과 secret을 checkpoint로 저장한다.
        mat_to_save = np.zeros((len(Y), len(Y)+self.N)).astype(int)
        mat_to_save[:len(X), :self.N] = X
        mat_to_save[-1, 0] = int(np.round(self.start))
        mat_to_save[:, self.N:] = Y
        np.save(self.matrix_filename, mat_to_save)

    def generate(self):
        # checkpoint가 있으면 이어서 돌리고, 없으면 새 embedding을 시작한다.
        if os.path.isfile(self.matrix_filename):
            secret_Ap = np.load(self.matrix_filename)
            secret, Ap = secret_Ap[:1, :self.N], secret_Ap[:, self.N:]
            self.start = secret_Ap[-1, 0]
        else:
            secret, Ap = self.get_Kannans_embedding()

        fplll_Ap = IntegerMatrix.from_matrix(Ap.tolist())
        M = GSO.Mat(fplll_Ap, float_type=self.float_type, update=True)
        BKZ_Obj = BKZ2(M)
        bkz_params = BKZ.EasyParam(self.block_size, delta=self.delta, max_time=MAX_TIME)
        start_time = None
        # 한 번의 BKZ 실행이 너무 오래 걸리면 다시 시도한다.
        while start_time is None or time() - start_time > MAX_TIME:
            start_time = time()
            try: 
                BKZ_Obj(bkz_params)
            except: 
                self.set_float_type(FLOAT_UPGRADE[self.float_type])
                logger.info(f'Error running bkz. Upgrading to float type {self.float_type}.')
                M = GSO.Mat(fplll_Ap, float_type=self.float_type, update=True)
                BKZ_Obj = BKZ2(M)
                BKZ_Obj(bkz_params)
            
            RAp = np.zeros((self.m+self.N+1, self.m+self.N+1), dtype=int)
            fplll_Ap.to_matrix(RAp)
            self.save_mat(secret, RAp)
            logger.info(f'Saved progress at {self.matrix_filename}')
            # 첫 행이 secret을 회복했는지 확인한다.
            if np.all(secret.flatten().astype(bool) == RAp[0, :self.N].astype(bool)):
                logger.info(f'Found secret for {self.matrix_filename}')
                break

        # 결과를 결과 파일에 누적하고, 다음 반복용 checkpoint도 다시 만든다.
        results = pickle.load(open(self.results_path, 'rb'))
        if type(results) != dict:
            results = results.__dict__
        logger.info(f'real secret = {secret.flatten()}. solved secret = {(RAp[0, :self.N]/self.weight).astype(int)}')
        success = np.all(secret.flatten().astype(bool) == RAp[0, :self.N].astype(bool))
        results[(self.expNum, self.hamming)] += (time() - self.start, success)
        pickle.dump(results, open(self.results_path, 'wb'))

        newSecret, newAp = self.get_Kannans_embedding()
        self.save_mat(newSecret, newAp)
        # export.py가 기대하는 형태에 맞추기 위해 secret을 패딩한다.
        ret_secret = np.zeros((1, self.m+self.N+1), dtype=int)
        ret_secret[0,:self.N] = secret
        return RAp, ret_secret.T
    
    def get_Kannans_embedding(self):
        # Kannan embedding은 secret recovery를 BKZ 문제로 바꾸는 표준 구성이다.
        N, m, Q = self.N, self.m, self.Q
        rng = np.random.RandomState(self.seed + [int(time())])
        A = rng.randint(0, Q, size=(m, N), dtype=np.int64)
        assert (np.min(A) >= 0) and (np.max(A) < Q)
        e = rng.normal(0, self.sigma, size = m).round()
        b = ((A@self.s).flatten() + e).astype(int) % Q
        # 이 시점부터 걸린 시간을 결과 기록에 사용한다.
        self.start = time()

        Ap = np.identity(m+N+1, dtype = int)
        Ap[:N, :N] *= self.weight
        Ap[:N, N:m+N] = A.T
        Ap[N:m+N, N:m+N] *= Q
        Ap[m+N, N:m+N] = b

        return self.s.reshape((1, N)), Ap

class RA_Rb():
    def __init__(self, params):
        # Ab 단계는 이전 단계에서 생성된 (R, b) prefix를 다시 읽는다.
        self.init_b(params)
        self.counter = 0
        self.index = 0
        self.seekpos = 0
        self.path = os.path.join(params.reload_data, 'data.prefix')
        self.rnorm = params.rnorm
        self.longtype = np.log2(params.Q) > 30
        # 이 단계는 파일 seek 위치를 공유하므로 병렬 worker를 허용하지 않는다.
        assert params.num_workers == 1
        assert os.path.isfile(self.path)
        self.reload_size = min(params.reload_size, 10000000 // self.N)
        self.loaded_X_R = []
        self.diff = None

    def init_b(self, params):
        # secret과 tiny 데이터 관련 부가 정보를 준비한다.
        self.N, self.Q, self.m = params.N, params.Q, params.m
        self.tiny1, self.tiny2 = params.tiny1, params.tiny2
        self.sigma = params.sigma
        # load secret (create if not exist)
        # secret_dir에 기존 secret이 있으면 그걸 그대로 쓴다.
        if os.path.isfile(os.path.join(params.secret_dir, 'secret.npy')):
            self.s = np.load(os.path.join(params.secret_dir, 'secret.npy'))
            if self.tiny1 or self.tiny2:
                self.tiny_b = np.load(os.path.join(params.secret_dir, 'orig_b.npy'))
            logger.info(f'Loaded secret (and b, if tiny) from {params.secret_dir}')
        else:
            # 없으면 요청된 secret_type에 맞춰 새 secret을 생성한다.
            secret_size = (self.N, params.num_secret_seeds * (params.max_hamming-params.min_hamming+1))
            if params.secret_type == "ternary":
                logger.info(f'Creating ternary secret')
                self.s = np.random.choice([-1,1], size = secret_size)
            elif params.secret_type == "gaussian":
                logger.info(f'Creating gaussian secret')
                self.s = np.random.normal(0, params.sigma, size = secret_size).round().astype(int)
            elif params.secret_type == "binomial":
                logger.info(f'Creating binomial secret')
                self.s = np.random.binomial(params.gamma, 0.5, secret_size) - np.random.binomial(params.gamma, 0.5, secret_size)
            else:
                logger.info(f'Creating binary secret')
                self.s = np.ones(secret_size).astype(int)
            # 해밍 가중치별로 원하는 개수만큼 0을 만들어 분포를 맞춘다.
            for h in range(params.min_hamming, params.max_hamming + 1):
                for seed_id in range(params.num_secret_seeds):
                    column = (h-params.min_hamming) * params.num_secret_seeds + seed_id
                    nonzeros = np.where(self.s[:, column] != 0)[0]
                    idxs = np.random.choice(nonzeros, size = len(nonzeros)-h, replace=False)
                    self.s[idxs, column] = 0
                    assert h == sum(self.s[:, column] != 0)
        np.save(os.path.join(params.dump_path, 'secret.npy'), self.s)

        # tiny 단계면 원본 A와 b를 다시 저장해 재사용한다.
        if self.tiny1 or self.tiny2:
            self.tiny_A = np.load(params.orig_A_path)
            np.save(os.path.join(params.dump_path, 'orig_A.npy'), self.tiny_A)
            if not os.path.isfile(params.secret_dir):
                # generate the e in the original b=A*s+e
                err_shape = (self.tiny_A.shape[0], self.s.shape[1])
                tiny_e = np.random.normal(0, params.sigma, size = err_shape).round().astype(int)
                if params.secret_type == "binomial":
                    tiny_e = np.random.binomial(params.gamma, 0.5, err_shape) - np.random.binomial(params.gamma, 0.5, err_shape)
                # Will be used by Rb in self.generate(), so that the distribution and dependence of vars are correct
                self.tiny_b = ((self.tiny_A @ self.s) + tiny_e) % self.Q
            np.save(os.path.join(params.dump_path, 'orig_b.npy'), self.tiny_b)

    def load_chunk(self):
        # data.prefix 파일을 한 번에 다 읽지 않고 chunk 단위로 불러온다.
        logger.info(f"Loading data from {self.path} ... seekpos {self.seekpos}")
        data, endfile = [], False
        with io.open(self.path, mode="r", encoding="utf-8") as f:
            f.seek(self.seekpos, 0)
            while len(data) < self.reload_size:
                A, RT = [], []
                for j in range(self.m):
                    line = f.readline()
                    if not line:
                        endfile = True
                        break
                    a, r = line.rstrip().split(";")
                    A.append(np.array(a.split()).astype(int))
                    RT.append(np.array(r.split()).astype(int))
                if len(A) == self.m:
                    data.append((np.array(A), np.array(RT).T))
                if endfile:
                    break
            self.seekpos = f.tell()

        self.loaded_X_R = data
        logger.info(f"Loaded {len(data)} matrices from the disk. seekpos {self.seekpos}")

    def generate(self):
        # 로드된 chunk를 순서대로 꺼내서 (RA, Rb)를 만든다.
        if self.index == len(self.loaded_X_R):
            self.load_chunk()
            if len(self.loaded_X_R) == 0:
                return None
            self.index = 0
        X, R = self.loaded_X_R[self.index]
        if self.tiny1:
            # tiny1은 저장된 인덱스 조합을 다시 A/b 행으로 풀어쓴다.
            A, b = np.zeros((self.N, self.N), dtype=int), np.zeros((self.N, self.s.shape[1]), dtype=int)
            for j in range(self.N):
                for k in X[j]:
                    A[j] += self.tiny_A[k]
                    b[j] += self.tiny_b[k]
        elif self.tiny2:
            # tiny2는 저장된 인덱스를 그대로 tiny_A / tiny_b에 적용한다.
            idxs = list(X.flatten())
            A = self.tiny_A[idxs]
            b = self.tiny_b[idxs]
        else: # not supporting binomial if not tiny
            # 일반 Ab 단계는 현재 chunk의 A에 secret과 noise를 곱해 b를 다시 만든다.
            A = X
            e = np.random.normal(0, self.sigma, size = self.s.shape).round()
            b = (A @ self.s + e).astype(int)

        R = R[[i for i in range(len(R)) if set(R[i]) != {0}]]
        if self.rnorm != 1:
            R = R[[i for i in range(len(R)) if np.linalg.norm(R[i])/self.Q < self.rnorm]]
        if self.longtype:
            R = R.astype(np.longdouble)
            RA = ((R//10000)@(A*10000%self.Q) + (R%10000)@A) % self.Q
            Rb = ((R//10000)@(b*10000%self.Q) + (R%10000)@b) % self.Q
        else:
            RA = (R@A) % self.Q
            Rb = (R@b) % self.Q

        _RA, _Rb = RA.copy(), Rb.copy()
        _RA[_RA>self.Q//2] -= self.Q
        _Rb[_Rb>self.Q//2] -= self.Q
        # diff는 secret 복원 품질을 추적하기 위한 누적 오차 벡터다.
        if self.diff is None:
            self.diff = (_RA@self.s - _Rb).astype(int)
        else:
            self.diff = np.concatenate([self.diff, (_RA@self.s - _Rb).astype(int)])

        self.index += 1
        self.counter += 1
        if self.counter % 100000 == 0:
            logger.info(f"Processed {self.counter} equations")
        return RA.astype(int), Rb.astype(int)
