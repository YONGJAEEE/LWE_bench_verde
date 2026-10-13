# Benchmarking t-version → Verde bundle

This directory is self-contained and was created without modifying either
source checkout. It contains:

- `benchmarking/`: the t-version-aware benchmarking generator.
- `verde/`: the Verde training code with t-version recovery support.
- `scripts/generate_and_convert.py`: runs the generator and converts its
  output into the Verde data layout.
- `scripts/run_benchmark_h10.sh`: foreground/background Hamming-10 `t_6`
  launcher with explicit `BUNDLE`, active `PYTHON`, and log setup.
- `scripts/run_verde_train.sh`: a small training launcher.

## Generate and convert

The generator expects a preprocessed benchmarking directory containing
`data.prefix` (or the corresponding input files). It creates `t_0` through
`t_7`, and the wrapper converts all eight versions automatically:

```bash
cd /Users/iyongjae/Documents/GitHub/LWE_bench_verde/verde_t_version_bundle
python3 scripts/generate_and_convert.py \
  --processed_dump_path /path/to/benchmark/preprocessed \
  --dump_path /path/to/benchmark/output \
  --verde_output_root /path/to/verde/ready \
  --hamming 3 \
  --secret_seed 0
```

For the requested Hamming-10, `t_version=6` run, set the processed dump path
to an absolute path. The launcher writes the complete command output to
`logs/benchmark_h10_t6.log` before returning in background mode:

```bash
PROCESSED_DUMP_PATH=/absolute/path/to/benchmark/preprocessed \
  scripts/run_benchmark_h10.sh
```

Use `FOREGROUND=1` when the calling shell should wait for completion. The
benchmark generator still creates `t_0` through `t_7`; only `t_6` is converted
by this launcher. The expected Verde directory is
`generated/verde_ready_h10/t_6`.

To convert an already-generated benchmark directory, use
`--convert_only --t_version 6` (or omit `--t_version` for all versions):

```bash
python3 scripts/generate_and_convert.py \
  --convert_only \
  --dump_path /path/to/benchmark/output \
  --verde_output_root /path/to/verde/ready \
  --hamming 3 \
  --secret_seed 0 \
  --t_version 6
```

Each output directory such as `verde/ready/t_6` contains `train.prefix`,
`test.prefix`, `orig_A.npy`, `orig_b.npy`, `secret.npy`, and `params.pkl`,
which are the files expected by Verde. `secret.npy` is an 80-vector padded
model guess with a zero prefix; Verde assembles the fixed one-hot prefix from
`t_version` during recovery.

## Train Verde

```bash
scripts/run_verde_train.sh /path/to/verde/ready/t_6 6 \
  --cpu true \
  --compile false \
  --epochs 20 \
  --check_secret_every 1 \
  --dump_path /path/to/verde/run
```

For the legacy non-split dataset, use `--t_version -1`. For t-version data,
use the matching `t_i` directory and `--t_version i`.
