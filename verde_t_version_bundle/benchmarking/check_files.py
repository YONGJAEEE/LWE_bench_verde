import os
import numpy as np

data_path = './data/benchmark_paper_data/n80_logq7/origA_n80_logq7_2/binary_secrets_h5_6'
parent = os.path.dirname(data_path)

print('Data path:', data_path)
print('Parent dir:', parent)
print()

# Check test_A.npy
test_A_path = os.path.join(parent, 'test_A.npy')
print(f'test_A.npy exists: {os.path.exists(test_A_path)}')
if os.path.exists(test_A_path):
    test_A = np.load(test_A_path)
    print(f'test_A shape: {test_A.shape}')
    print(f'test_A length: {len(test_A)}')
else:
    print(f'test_A.npy not found at: {test_A_path}')

print()

# Check test_b_5_0.npy
test_b_path = os.path.join(data_path, 'test_b_5_0.npy')
print(f'test_b_5_0.npy exists: {os.path.exists(test_b_path)}')
if os.path.exists(test_b_path):
    test_b = np.load(test_b_path)
    print(f'test_b shape: {test_b.shape}')
    print(f'test_b length: {len(test_b)}')
else:
    print(f'test_b_5_0.npy not found at: {test_b_path}')

print()

# Check what files exist in parent
print('Files in parent directory:')
if os.path.exists(parent):
    files = [f for f in os.listdir(parent) if f.endswith('.npy')]
    for f in sorted(files):
        print(f'  - {f}')

print()

# Check what test_b files exist
print('test_b files in data_path:')
if os.path.exists(data_path):
    files = [f for f in os.listdir(data_path) if f.startswith('test_b_') and f.endswith('.npy')]
    for f in sorted(files):
        print(f'  - {f}')

