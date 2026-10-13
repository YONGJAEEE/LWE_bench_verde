import os
import numpy as np

# Check data.prefix
prefix_path = './data/benchmark_paper_data/n80_logq7/data.prefix'
print(f'data.prefix exists: {os.path.exists(prefix_path)}')
if os.path.exists(prefix_path):
    size = os.path.getsize(prefix_path)
    print(f'data.prefix size: {size} bytes')
    if size > 0:
        # Try to read first few lines
        with open(prefix_path, 'r') as f:
            lines = [f.readline() for _ in range(3)]
            print(f'First 3 lines (first 100 chars each):')
            for i, line in enumerate(lines):
                if line:
                    print(f'  Line {i+1}: {line[:100]}')
                else:
                    print(f'  Line {i+1}: (empty)')
                    break

print()

# Check generated files
data_dir = './data/benchmark_paper_data/n80_logq7/origA_n80_logq7_2'
files_to_check = ['train_A.npy', 'test_A.npy', 'orig_A.npy']
for fname in files_to_check:
    fpath = os.path.join(data_dir, fname)
    if os.path.exists(fpath):
        arr = np.load(fpath)
        print(f'{fname}: shape={arr.shape}, size={arr.size}')
    else:
        print(f'{fname}: NOT FOUND')

print()

# Check if there's any data in the prefix file
if os.path.exists(prefix_path) and os.path.getsize(prefix_path) > 0:
    print('data.prefix has content. The issue might be in the data reading process.')
else:
    print('data.prefix is empty or missing. Need to regenerate data.')

