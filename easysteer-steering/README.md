# EasySteer Steering Wrapper

This directory is a small wrapper around the [EasySteer](EasySteer) submodule checkout.

It lets you run [easysteer-steering.py](easysteer-steering.py) from this directory instead of having to `cd` into `EasySteer/`. The script resolves the custom `vllm-steer` source tree and the steering vector file relative to this wrapper, so the working directory no longer needs to be `EasySteer/`.

## Files

- [easysteer-steering.py](easysteer-steering.py): demo script that compares baseline output with a "happy" steering vector.
- [EasySteer](EasySteer): EasySteer checked out as a git submodule, including `vllm-steer/` and `vectors/`.
- [pyproject.toml](pyproject.toml): local `uv` project used when running the wrapper from this directory.

## Cloning

If you are starting from a fresh clone of the top-level repo, make sure submodules are present:

```bash
git submodule update --init --recursive
```

## Requirements

- Linux with a CUDA-capable NVIDIA GPU visible to PyTorch
- `uv`
- Python 3.10 recommended to match `EasySteer/.venv`

## One-Time EasySteer Setup

If `EasySteer/.venv` is not ready yet, set it up inside the EasySteer submodule first:

```bash
cd EasySteer
uv venv --python 3.10
source .venv/bin/activate

cd vllm-steer
export VLLM_PRECOMPILED_WHEEL_COMMIT=95c0f928cdeeaa21c4906e73cee6a156e1b3b995
VLLM_USE_PRECOMPILED=1 uv pip install --editable . -v

cd ..
uv pip install --editable . -v
```

Then return to this wrapper directory:

```bash
cd ..
```

## Running From This Directory

The script defaults `CUDA_VISIBLE_DEVICES` to `0` when the variable is unset.

If you want a different GPU, override it in the shell before running:

```bash
export CUDA_VISIBLE_DEVICES=1
```

Preferred if you want to reuse the existing `EasySteer/.venv` without letting `uv` resync it:

```bash
EasySteer/.venv/bin/python easysteer-steering.py
```

If you want to use `uv` while still targeting the already-activated EasySteer environment:

```bash
source EasySteer/.venv/bin/activate
uv run --active --no-sync easysteer-steering.py
```

Place `--active` before the script name so `uv` handles it as a `uv run` option.

If you run the wrapper as its own local `uv` project:

```bash
uv run easysteer-steering.py
```

That command creates or uses a separate `.venv` in this directory. It does not reuse `EasySteer/.venv`, so it may need to install large dependencies again. `uv` can reuse cached downloads, but expect another environment solve/install and potentially many gigabytes of packages such as `torch`, `triton`, and related CUDA wheels.

If you already have `EasySteer/.venv` prepared, prefer `--no-sync` or call the interpreter directly to avoid `uv` mutating that environment.

## Runtime Knobs

The script supports a few environment variables:

- `CUDA_VISIBLE_DEVICES`: choose which GPU is visible; if unset, the script defaults to `0`
- `EASYSTEER_GPU_MEMORY_UTILIZATION`: float in `(0, 1]` to override the automatic memory cap
- `EASYSTEER_MAX_MODEL_LEN`: `-1` for model default, or a positive integer

## Expected Behavior

The script loads `Qwen/Qwen2.5-1.5B-Instruct`, runs one baseline generation, then runs a second generation with the bundled `happy_diffmean.gguf` steering vector applied across layers 10 through 25.
