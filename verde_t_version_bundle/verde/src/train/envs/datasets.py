# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
import io
import sys
import time
import numpy as np
from logging import getLogger
logger = getLogger()


T_VERSION_COUNT = 8
T_VERSION_ACTUAL = 6


class EnvDataset(Dataset):
    def __init__(self, params, env, datatype):
        super().__init__()
        
        self.env = env
        self.train = datatype == 'train'
        self.env_base_seed = params.env_base_seed
        self.batch_size = params.batch_size
        self.add_unred_perc = params.add_unred_perc
        self.secret_col = params.secret_col
        self.data_cols, self.dense_cols = params.data_cols, params.dense_cols

        # batching
        self.num_workers = params.num_workers
        self.global_rank = params.global_rank
        self.local_rank = params.local_rank
        self.n_gpu_per_node = params.n_gpu_per_node

        if self.train:
            self.reload_size = params.reload_size
            self.path = params.train_path
        else:
            self.reload_size = params.eval_size
            self.path = params.test_path

        self.batch_load = params.batch_load
        self.shuffle = params.shuffle
        self.seekpos = 0
        self.basepos = 0
        self.nextpos = 0
        
        assert params.reload_size > 0
        assert params.num_workers in [0, 1]
        assert datatype in ["train", "test"]

        self.Q = params.Q
        self.load_chunk()
        if self.train:
            self.size = 1 << 60
        else:
            self.size = len(self.data)
    
        # load the original tiny dataset if available
        self.orig_A, self.orig_b = None, None
        if os.path.isfile(os.path.join(params.reload_data, 'orig_A.npy')):
            self.orig_A = np.asarray(np.load(os.path.join(params.reload_data, 'orig_A.npy')))
            raw_orig_b = np.asarray(np.load(os.path.join(params.reload_data, 'orig_b.npy')))
            if self.orig_A.ndim != 2:
                raise ValueError(f'orig_A must be a 2-D integer array; got {self.orig_A.shape}.')
            if raw_orig_b.ndim == 1:
                raw_orig_b = raw_orig_b.reshape(-1, 1)
            if raw_orig_b.ndim != 2 or self.secret_col < 0 or self.secret_col >= raw_orig_b.shape[1]:
                raise ValueError(
                    f'orig_b must have secret column {self.secret_col}; got shape {raw_orig_b.shape}.'
                )
            if raw_orig_b.shape[0] != self.orig_A.shape[0]:
                raise ValueError(
                    f'orig_A/orig_b row mismatch: {self.orig_A.shape[0]} vs {raw_orig_b.shape[0]}.'
                )
            self.orig_b = raw_orig_b[:, self.secret_col]
            if self.data_cols is not None:
                if len(self.dense_cols) > 0:
                    self.orig_b -= np.sum(self.orig_A[:, self.dense_cols], axis=1)
                    self.orig_b %= params.Q
                    self.orig_A[:, self.dense_cols] *= -1
                    self.orig_A[:, self.dense_cols] %= params.Q
                self.orig_A = self.orig_A[:, self.data_cols]

            for name, array in (('orig_A', self.orig_A), ('orig_b', self.orig_b)):
                if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(array.dtype, np.integer):
                    raise ValueError(f'{name} must be an integer array; got {array.dtype}.')
                if not np.all(np.isfinite(array)):
                    raise ValueError(f'{name} contains NaN or Inf.')
                if np.any((array < 0) | (array >= params.Q)):
                    raise ValueError(f'{name} contains values outside [0, Q).')
            if self.orig_A.shape[1] != params.N:
                raise ValueError(
                    f'orig_A has {self.orig_A.shape[1]} columns but params.N={params.N}.'
                )

            secret_for_check = np.asarray(params.secret).reshape(-1)
            if np.issubdtype(secret_for_check.dtype, np.bool_) or not np.issubdtype(secret_for_check.dtype, np.integer):
                raise ValueError(f'secret must be an integer vector; got {secret_for_check.dtype}.')
            if not np.all(np.isfinite(secret_for_check)):
                raise ValueError('secret contains NaN or Inf.')
            t_version = getattr(params, 't_version', -1)
            if isinstance(t_version, bool) or not isinstance(t_version, (int, np.integer)):
                raise ValueError('t_version must be -1 or an integer in the range 0..7.')
            if t_version != -1:
                if t_version not in range(T_VERSION_COUNT):
                    raise ValueError('t_version must be -1 or an integer in the range 0..7.')
                if params.N != 80 or len(secret_for_check) not in (72, 80):
                    raise ValueError(
                        't_version dataset validation requires N=80 and a '
                        '72- or 80-element secret guess.'
                    )
                t = np.zeros(T_VERSION_COUNT, dtype=int)
                t[t_version] = 1
                s1_guess = (
                    secret_for_check[8:]
                    if len(secret_for_check) == 80
                    else secret_for_check
                )
                secret_for_check = np.concatenate([t, s1_guess])
                if t_version == T_VERSION_ACTUAL:
                    err = (self.orig_A @ secret_for_check - self.orig_b) % params.Q
                    err[err > params.Q // 2] -= params.Q
                    residual_std = float(np.std(err))
                    threshold = 2 * params.sigma
                    if not np.isfinite(residual_std) or residual_std >= threshold:
                        raise ValueError(
                            f't-version {t_version} residual validation failed: '
                            f'std={residual_std:.6f}, threshold={threshold:.6f}.'
                        )
                    logger.info(
                        'Validated correct t-version %d: residual std=%.6f < threshold=%.6f.',
                        t_version, residual_std, threshold,
                    )
                else:
                    logger.info(
                        't-version %d is an intentional incorrect hypothesis; '
                        'skipping correct-secret residual assertion.', t_version,
                    )
            else:
                if len(secret_for_check) != self.orig_A.shape[1]:
                    raise ValueError(
                        f'Legacy secret length {len(secret_for_check)} does not match '
                        f'orig_A columns {self.orig_A.shape[1]}.'
                    )
                err = (self.orig_A @ secret_for_check - self.orig_b) % params.Q
                err[err > params.Q // 2] -= params.Q
                residual_std = float(np.std(err))
                threshold = 2 * params.sigma
                if not np.isfinite(residual_std) or residual_std >= threshold:
                    raise ValueError(
                        f'Legacy residual validation failed: std={residual_std:.6f}, '
                        f'threshold={threshold:.6f}.'
                    )

    def load_chunk(self):
        rng = np.random.RandomState([self.env_base_seed, int(time.time())])
        self.basepos = self.nextpos
        logger.info(
            f"Loading data from {self.path} ... seekpos {self.seekpos}, basepos {self.basepos}"
            )
        endfile = False
        with io.open(self.path, mode="r", encoding="utf-8") as f:
            f.seek(self.seekpos, 0)
            lines, i = [], 0
            while len(lines) < self.reload_size:
                line = f.readline()
                if not line:
                    endfile = True
                    break
                if i % self.n_gpu_per_node == self.local_rank:
                    row = line.split(';')
                    if len(row) == 2:
                        x = np.array(row[0].split()).astype(int)
                        y = int(row[1].split()[self.secret_col])
                        if self.data_cols is not None:
                            if len(self.dense_cols) > 0:
                                y -= sum(x[self.dense_cols])
                                y %= self.Q
                                x[self.dense_cols] *= -1
                                x[self.dense_cols] %= self.Q
                            x = x[self.data_cols]
                        y = np.array([y]).astype(int)
                        assert len(x) >= 1 and len(y) == 1
                        lines.append((x,y))
                i += 1
            self.seekpos = 0 if endfile else f.tell()

        if self.shuffle:
            rng.shuffle(lines)
        self.data = lines
        self.nextpos = self.basepos + len(lines)
        logger.info(
            f"Loaded {len(self.data)} equations from the disk. seekpos {self.seekpos}, nextpos {self.nextpos}"
            )
        if len(self.data) == 0:
            raise RuntimeError(f'No valid equations were loaded from {self.path}.')

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        x, y = zip(*elements)
        x_ar = []
        y_ar = []
        for e1, e2 in zip(x,y):
            x_ar.append(np.array(e1))
            y_ar.append(np.array(e2))
    
        # Fix the length to be the actual batch size.
        x = x_ar[:self.batch_size]
        y = y_ar[:self.batch_size]
   
        # Distinguish the different equations
        int_len = len(self.env.input_encoder.write_int(0))
        nb_eqs = [self.env.code_class(xi, yi, int_len) for xi, yi in zip(x, y)]

        # Pad 
        x = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in x]
        y = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in y]
        x, x_len = self.env.batch_sequences(x)
        y, y_len = self.env.batch_sequences(y)
        return (x, x_len), (y, y_len), torch.LongTensor(nb_eqs)

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if hasattr(self.env, "rng"):
            return
        if self.train:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            ttime = int(time.time())
            self.env.rng = np.random.RandomState(
                [worker_id, self.global_rank, self.env_base_seed, ttime]
            )
            logger.info(
                f"Initialized random generator for worker {worker_id}, with seed "
                f"{[worker_id, self.global_rank, self.env_base_seed, ttime]} "
                f"(base seed={self.env_base_seed})."
            )
        else:
            self.env.rng = np.random.RandomState(0)

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0)
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def __getitem__(self, index):
        """
        Return a train / test sample.
        """
        self.init_rng()
        idx = index
        if self.train:
            if self.batch_load:
                if index >= self.nextpos:
                    self.load_chunk()
                idx = index - self.basepos
            else:
                index = self.env.rng.randint(len(self.data))
                idx = index
        x, y = self.data[idx]
        if self.env.rng.randint(100) < self.add_unred_perc:
            rand_comb = self.env.rng.choice(len(self.orig_A), size = 4, replace=False)
            x = np.sum(self.orig_A[rand_comb], axis=0) % self.Q
            y = np.array([sum(self.orig_b[rand_comb]) % self.Q])
        return self.tokenize(x, True), self.tokenize(y, False)
    
    def getbatchA(self, num_samples):
        return [x for x, _ in self.data[:num_samples]]
    
    def tokenize(self, x, is_input):
        if is_input:
            x_encoded = self.env.input_encoder.encode(x)
        else:
            x_encoded = self.env.output_encoder.encode(x)
        return x_encoded

class ReuseDataset(EnvDataset):
    def __init__(self, params, env, datatype='train'):
        super().__init__(params, env, datatype)

        self.samples = []
        
        if self.train:
            self.size = params.num_reuse_samples
    
    def populate_dataset(self):
        self.init_rng()
        self.samples = [self.generate_sample() for _ in range(self.size_) ]
        self.ptr = 0
    
    def __getitem__(self, index):
        if len(self.samples) == 0:
            start = time.time()
            self.populate_dataset()
            print(f"Time to populate dataset: {time.time() - start}")
        
        sample = self.samples[self.ptr%self.size_]
        self.ptr +=1

        return sample

def create_dataloader(params, env, dataset_type):
    """
    Create a dataset for this environment.
    """
    dataset = EnvDataset(params, env, dataset_type)
    logger.info(f"Creating {dataset_type} dataloader...")
    return DataLoader(
        dataset,
        timeout = (0 if params.num_workers == 0 else 7200),
        batch_size = params.batch_size,
        num_workers = (params.num_workers if dataset_type == 'train' else 1),
        shuffle = False,
        collate_fn = dataset.collate_fn
    )
