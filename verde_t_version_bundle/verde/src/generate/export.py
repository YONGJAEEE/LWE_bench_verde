# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import io
import sys
import ast
import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch import nn

from ..utils import to_cuda, timeout, TimeoutError


logger = getLogger()


class Generator(object):
    def __init__(self, params, gen):
        """
        Initialize trainer.
        """
        # params는 전체 생성 과정의 공통 설정값이고,
        # gen은 step별 실제 샘플러다.
        self.params = params
        self.gen = gen
        self.print_cycle = 500 if params.step == "Ab" else 1
        
        # epoch / iteration size
        self.epoch_size = params.epoch_size
        assert self.epoch_size > 0

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.timeout_count = 0
        self.total_count = 0
        self.stats = { "processed_e": 0 }
        self.last_time = time.time()

        # export 결과는 prefix 파일로 쌓는다.
        export_path_prefix = os.path.join(params.dump_path, "data.prefix")
        if params.step == 'Ab':
            export_path_prefix = os.path.join(params.dump_path, "test.prefix")
        self.file_handler_prefix = io.open(export_path_prefix, mode="a", encoding="utf-8")
        logger.info(f"Data will be stored in: {export_path_prefix} ...")

    def iter(self):
        """
        End of iteration.
        """
        # 한 샘플 생성이 끝났으니 내부 카운터를 갱신한다.
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        # print_cycle마다만 로그를 찍어서 출력이 너무 많아지지 않게 한다.
        if self.n_total_iter % self.print_cycle != 0:
            return

        s_iter = "%7i - " % self.n_total_iter
        s_stat = " || ".join(
            [
                "{}: {:7.4f}".format(k.upper().replace("_", "-"), np.mean(v))
                for k, v in self.stats.items()
                if type(v) is list and len(v) > 0
            ]
        )
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # 이번 구간의 처리 속도를 계산한다.
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} equations/s - ".format(
            self.stats["processed_e"] * 1.0 / diff,
        )
        self.stats["processed_e"] = 0
        self.last_time = new_time

        # 속도와 통계를 한 줄로 남긴다.
        logger.info(s_iter + s_speed + s_stat)

    def end_epoch(self):
        """
        End the epoch. 
        """
        # Ab 단계는 생성 중에 모아둔 diff를 별도 파일로 저장한다.
        if self.params.step == 'Ab':
            np.save(os.path.join(self.params.dump_path, 'diff.npy'), self.gen.diff)
            logger.info(f'Saved diff at {os.path.join(self.params.dump_path, "diff.npy")}')

    def export_data(self):
        """
        Export data to the disk.
        """
        # sampleGen이 성공할 때까지 재시도한다.
        while True:
            try:
                sample = self.gen.generate()
                break
            except TimeoutError:
                logger.info(f'Timeout, count: {self.timeout_count}')
                self.timeout_count += 1
                if self.timeout_count >= self.params.max_timeout_count:
                    # 너무 오래 막히면 이번 샘플은 포기한다.
                    sample = None
                    break
            except Exception as e:
                logger.info(f'An exception happened in the generator: {e}')
                continue
        self.timeout_count = 0

        if sample is None:
            return None
        else:
            X, Y = sample
            assert X.shape[0] == Y.shape[0]
            for i in range(X.shape[0]):
                # 한 행씩 사람이 읽을 수 있는 prefix 형식으로 저장한다.
                prefix1_str = " ".join(X[i].astype(str))
                prefix2_str = " ".join(Y[i].astype(str))
                self.file_handler_prefix.write(f"{prefix1_str} ; {prefix2_str}\n")
                self.total_count += 1

                # Ab 단계는 앞부분을 train, 뒷부분을 test처럼 나눠 쓰기 위해
                # 일정 개수 이후 파일을 교체한다.
                if self.total_count == 10000 and self.params.step == 'Ab':
                    self.file_handler_prefix.flush()
                    export_path_prefix = os.path.join(self.params.dump_path, "train.prefix")
                    self.file_handler_prefix = io.open(export_path_prefix, mode="a", encoding="utf-8")

            # 버퍼에 남은 내용을 바로 디스크에 밀어 넣는다.
            self.file_handler_prefix.flush()

        # 통계에는 처리한 equation 수와 토큰 수를 반영한다.
        self.n_equations += 1
        self.stats["processed_e"] += 2*self.params.N
        return True
