import numpy as np
import os

# 생성된 데이터 확인
print("=" * 80)
print("생성된 데이터에서 시크릿 사용 확인")
print("=" * 80)

# 1. orig_b.npy 확인
orig_b_path = 'data/benchmark_paper_data/n80_logq7/origA_n80_logq7_2/orig_b.npy'
orig_b = np.load(orig_b_path)
print(f"\n1. orig_b.npy")
print(f"   Shape: {orig_b.shape}")
if len(orig_b.shape) == 2:
    print(f"   -> {orig_b.shape[1]}개의 시크릿에 대한 b 데이터 포함")
elif len(orig_b.shape) == 3:
    print(f"   -> {orig_b.shape[2]}개의 시크릿에 대한 b 데이터 포함 (RLWE)")

# 2. secret.npy 확인
secret_path = 'data/benchmark_paper_data/n80_logq7/origA_n80_logq7_2/binary_secrets_h5_6/secret.npy'
secret = np.load(secret_path)
print(f"\n2. secret.npy (생성된 디렉토리)")
print(f"   Shape: {secret.shape}")
print(f"   -> {secret.shape[1]}개의 시크릿 포함")

# 3. 개별 시크릿 파일들 확인
secret_dir = 'data/benchmark_paper_data/n80_logq7/origA_n80_logq7_2/binary_secrets_h5_6'
secret_files = [f for f in os.listdir(secret_dir) if f.startswith('secret_') and f.endswith('.npy')]
secret_files.sort()

print(f"\n3. 개별 시크릿 파일들")
print(f"   총 {len(secret_files)}개의 개별 시크릿 파일:")
for f in secret_files:
    print(f"   - {f}")

# 4. orig_b 파일들 확인
orig_b_files = [f for f in os.listdir(secret_dir) if f.startswith('orig_b_') and f.endswith('.npy')]
orig_b_files.sort()

print(f"\n4. 개별 orig_b 파일들")
print(f"   총 {len(orig_b_files)}개의 orig_b 파일:")
for f in orig_b_files:
    print(f"   - {f}")

# 5. train_b 파일들 확인
train_b_files = [f for f in os.listdir(secret_dir) if f.startswith('train_b_') and f.endswith('.npy')]
train_b_files.sort()

print(f"\n5. train_b 파일들")
print(f"   총 {len(train_b_files)}개의 train_b 파일")

# 6. test_b 파일들 확인
test_b_files = [f for f in os.listdir(secret_dir) if f.startswith('test_b_') and f.endswith('.npy')]
test_b_files.sort()

print(f"\n6. test_b 파일들")
print(f"   총 {len(test_b_files)}개의 test_b 파일")

# 7. 요약
print("\n" + "=" * 80)
print("요약:")
print("=" * 80)
print(f"- secret.npy: {secret.shape[1]}개 시크릿")
print(f"- 개별 secret 파일: {len(secret_files)}개")
print(f"- 개별 orig_b 파일: {len(orig_b_files)}개")
print(f"- 개별 train_b 파일: {len(train_b_files)}개")
print(f"- 개별 test_b 파일: {len(test_b_files)}개")

if secret.shape[1] == len(secret_files) == len(orig_b_files) == len(train_b_files) == len(test_b_files):
    print(f"\n✓ 모든 {secret.shape[1]}개의 시크릿이 사용되었습니다!")
else:
    print(f"\n⚠ 일부 시크릿이 누락되었을 수 있습니다.")

