import numpy as np

data = np.load('data/benchmark_paper_data/n80_logq7/binary_secrets_h5_6/secret.npy')

print(f'Shape: {data.shape}')
print('\n' + '=' * 80)
print('Matrix:')
print('=' * 80)

# 행 번호와 함께 출력
for i, row in enumerate(data):
    row_str = ' '.join(f'{int(x):2d}' for x in row)
    print(f'[{i:2d}] {row_str}')

print('=' * 80)
print(f'\nTotal: {data.shape[0]} rows x {data.shape[1]} columns')

# 마지막 8개 행 (72-79)에서 1이 정확히 1개만 있는 시크릿 찾기
print('\n' + '=' * 80)
print('마지막 8개 행(72-79)에서 1이 정확히 1개만 있는 시크릿 찾기:')
print('=' * 80)

last_8_rows = data[72:80, :]  # 마지막 8개 행
ones_count = np.sum(last_8_rows, axis=0)  # 각 열(시크릿)별로 1의 개수

# 1이 정확히 1개인 시크릿 찾기
secrets_with_one = np.where(ones_count == 1)[0]

print(f'\n총 {len(secrets_with_one)}개의 시크릿이 마지막 8개 행에 1이 정확히 1개 있습니다.')
print(f'시크릿 인덱스: {secrets_with_one.tolist()}')

# 각 시크릿의 상세 정보 출력
print('\n상세 정보:')
print('-' * 80)
for secret_idx in secrets_with_one:
    secret_col = last_8_rows[:, secret_idx]
    one_position = np.where(secret_col == 1)[0][0]  # 1이 있는 위치 (0-7)
    actual_row = 72 + one_position  # 실제 행 번호
    print(f'시크릿 {secret_idx:2d}: 행 {actual_row:2d}에 1이 있음')
    print(f'  마지막 8개 행: {secret_col.tolist()}')
    print()

