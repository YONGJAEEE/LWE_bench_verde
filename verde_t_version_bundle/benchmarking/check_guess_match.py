#!/usr/bin/env python3
"""
full_guess가 실제 secret과 일치하는지 확인하는 스크립트
"""
import numpy as np
import pickle
import sys

def check_guess_match(guess_log_path, params_path=None, secret_path=None):
    """CSV의 마지막 full_guess와 실제 secret을 비교"""
    
    # CSV에서 마지막 full_guess 읽기
    import csv
    with open(guess_log_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            print("CSV 파일이 비어있습니다.")
            return
        
        last_row = rows[-1]
        full_guess_str = last_row['full_guess']
        t_version = int(last_row['t_version'])
        matched = last_row['matched'] == 'True'
        
        print(f"마지막 행 정보:")
        print(f"  t_version: {t_version}")
        print(f"  matched: {matched}")
        print(f"  full_guess: {full_guess_str[:100]}...")
        
        # full_guess를 배열로 변환
        full_guess = np.array([int(x) for x in full_guess_str.split(',')])
        print(f"  full_guess 길이: {len(full_guess)}")
        print(f"  full_guess 1의 개수: {np.sum(full_guess)}")
    
    # params.pkl에서 정보 읽기
    if params_path:
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
            print(f"\nParams 정보:")
            print(f"  N: {params.N}")
            print(f"  Q: {params.Q}")
            print(f"  sigma: {params.sigma}")
            print(f"  secret_type: {params.secret_type}")
            if hasattr(params, 't_version'):
                print(f"  t_version: {params.t_version}")
            if hasattr(params, 'data_path'):
                print(f"  data_path: {params.data_path}")
    
    # 실제 secret 로드 및 비교
    if secret_path:
        actual_secret = np.load(secret_path)
        print(f"\n실제 secret 정보:")
        print(f"  secret 길이: {len(actual_secret)}")
        print(f"  secret 1의 개수: {np.sum(actual_secret)}")
        print(f"  secret: {actual_secret}")
        
        # t_version이 설정된 경우, full_s 조립
        if t_version >= 0:
            t = np.zeros((8,), dtype=np.int64)
            t[t_version] = 1
            if len(actual_secret) == 72:
                full_actual_secret = np.concatenate([t, actual_secret])
            elif len(actual_secret) == 80:
                full_actual_secret = np.concatenate([t, actual_secret[8:]])
            else:
                full_actual_secret = actual_secret
        else:
            full_actual_secret = actual_secret
        
        print(f"\n비교 결과:")
        print(f"  full_actual_secret 길이: {len(full_actual_secret)}")
        print(f"  full_actual_secret: {full_actual_secret}")
        print(f"  full_guess: {full_guess}")
        
        if len(full_actual_secret) == len(full_guess):
            is_exact_match = np.array_equal(full_actual_secret, full_guess)
            diff_count = np.sum(full_actual_secret != full_guess)
            print(f"  정확히 일치: {is_exact_match}")
            print(f"  다른 비트 수: {diff_count}")
            if diff_count > 0:
                diff_indices = np.where(full_actual_secret != full_guess)[0]
                print(f"  다른 비트 인덱스: {diff_indices[:20]}...")  # 처음 20개만
        else:
            print(f"  길이가 다릅니다: {len(full_actual_secret)} vs {len(full_guess)}")
    
    # A와 b를 사용한 검증
    if params_path and hasattr(params, 'data_path'):
        import os
        A_path = os.path.join(os.path.dirname(params.data_path), "orig_A.npy")
        b_path = os.path.join(params.data_path, f"orig_b_{params.hamming}_{params.secret_seed}.npy")
        
        if os.path.exists(A_path) and os.path.exists(b_path):
            A = np.load(A_path, mmap_mode='r')
            b = np.load(b_path, mmap_mode='r')
            
            # 첫 번째 샘플만 사용
            A_sample = A[0] if len(A.shape) > 1 else A
            b_sample = b[0] if len(b.shape) > 1 else b
            
            print(f"\nA와 b를 사용한 검증:")
            print(f"  A shape: {A_sample.shape}")
            print(f"  b shape: {b_sample.shape}")
            
            # full_guess로 검증
            err_pred = (A_sample @ full_guess - b_sample) % params.Q
            err_pred[err_pred > params.Q // 2] -= params.Q
            err_std = np.std(err_pred)
            err_mean = np.mean(err_pred)
            err_max = np.max(np.abs(err_pred))
            
            print(f"  err_std: {err_std:.6f}")
            print(f"  2*sigma: {2*params.sigma:.6f}")
            print(f"  err_mean: {err_mean:.6f}")
            print(f"  err_max: {err_max:.6f}")
            print(f"  matched 조건 (err_std < 2*sigma): {err_std < 2*params.sigma}")
            
            # 실제 secret으로 검증
            if secret_path:
                err_actual = (A_sample @ full_actual_secret - b_sample) % params.Q
                err_actual[err_actual > params.Q // 2] -= params.Q
                err_actual_std = np.std(err_actual)
                err_actual_mean = np.mean(err_actual)
                err_actual_max = np.max(np.abs(err_actual))
                
                print(f"\n  실제 secret으로 검증:")
                print(f"  err_std: {err_actual_std:.6f}")
                print(f"  err_mean: {err_actual_mean:.6f}")
                print(f"  err_max: {err_actual_max:.6f}")
                print(f"  matched 조건 (err_std < 2*sigma): {err_actual_std < 2*params.sigma}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python check_guess_match.py <guess_log.csv> [params.pkl] [secret.npy]")
        sys.exit(1)
    
    guess_log_path = sys.argv[1]
    params_path = sys.argv[2] if len(sys.argv) > 2 else None
    secret_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    check_guess_match(guess_log_path, params_path, secret_path)
