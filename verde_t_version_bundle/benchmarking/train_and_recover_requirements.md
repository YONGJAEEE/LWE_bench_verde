# train_and_recover.py 실행에 필요한 인자와 파일

## 필수 인자 (Required Arguments)

1. **--data_path** (required)
   - b 파일들이 있는 디렉토리 경로
   - 예: `./data/benchmark_paper_data/n80_logq7/origA_n80_logq7_2/binary_secrets_h5_6`

2. **--secret_seed** (required, type=int)
   - 사용할 시크릿의 시드 번호 (0~9)
   - 예: `--secret_seed 0`

3. **--hamming** (required, type=int)
   - Hamming weight 값
   - 예: `--hamming 5` 또는 `--hamming 6`

## 필요한 파일들

### 1. params.pkl 파일 (하나만 있으면 됨)
다음 위치 중 하나에 있어야 함:
- `{data_path}/params.pkl`
- `{data_path}/secrets/params.pkl`
- `{data_path}/../params.pkl` (부모 디렉토리)

**내용**: N, Q, sigma, gamma, secret_type 정보

### 2. A 행렬 파일들 (data_path의 부모 디렉토리에 있어야 함)
- `{data_path의 부모}/train_A.npy` - 학습용 A 행렬
- `{data_path의 부모}/test_A.npy` - 테스트용 A 행렬  
- `{data_path의 부모}/orig_A.npy` - 원본 A 행렬 (secret recovery 검증용)

### 3. b 벡터 파일들 (data_path에 있어야 함)
- `{data_path}/train_b_{hamming}_{secret_seed}.npy` - 학습용 b
- `{data_path}/test_b_{hamming}_{secret_seed}.npy` - 테스트용 b
- `{data_path}/orig_b_{hamming}_{secret_seed}.npy` - 원본 b (secret recovery 검증용)

## 예시 명령어

```bash
python src/salsa/train_and_recover.py \
  --data_path ./data/benchmark_paper_data/n80_logq7/origA_n80_logq7_2/binary_secrets_h5_6 \
  --secret_seed 0 \
  --hamming 5 \
  --dump_path ./output \
  --exp_name test_run
```

## 현재 생성된 데이터 기준 예시

생성된 데이터 구조:
```
origA_n80_logq7_2/
├── train_A.npy          # 부모 디렉토리에 있음
├── test_A.npy           # 부모 디렉토리에 있음
├── orig_A.npy           # 부모 디렉토리에 있음
└── binary_secrets_h5_6/
    ├── params.pkl       # 여기에 있음
    ├── train_b_5_0.npy  # hamming=5, seed=0
    ├── test_b_5_0.npy
    ├── orig_b_5_0.npy
    ├── train_b_5_1.npy  # hamming=5, seed=1
    └── ...
```

실행 예시:
```bash
# hamming=5, seed=0인 시크릿으로 학습
python src/salsa/train_and_recover.py \
  --data_path ./data/benchmark_paper_data/n80_logq7/origA_n80_logq7_2/binary_secrets_h5_6 \
  --secret_seed 0 \
  --hamming 5

# hamming=6, seed=3인 시크릿으로 학습
python src/salsa/train_and_recover.py \
  --data_path ./data/benchmark_paper_data/n80_logq7/origA_n80_logq7_2/binary_secrets_h5_6 \
  --secret_seed 3 \
  --hamming 6
```

