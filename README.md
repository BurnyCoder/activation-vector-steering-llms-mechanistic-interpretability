# Steering Activation Vectors

This repo is a small workspace for trying two different styles of activation steering:

- [`steering-vectors-steering/`](steering-vectors-steering): a minimal `steering-vectors` demo using GPT-2 and a saved `.pt` steering vector
- [`easysteer-steering/`](easysteer-steering): a wrapper around an [EasySteer](easysteer-steering/EasySteer) submodule checkout for GPU-backed steering with a custom `vllm` tree

The two paths are intentionally separate. The small demo is simple and lightweight. The EasySteer path is much heavier, CUDA-dependent, and brings its own environment and upstream codebase.

## What This Repo Is For

Use this repo if you want to:

- see a minimal steering-vector example end to end
- compare that with a larger production-style steering stack built on EasySteer and `vllm`
- keep wrapper scripts, pretrained vectors, and upstream EasySteer code in one place

This repo is better thought of as a workspace than as one unified Python package.

## Which Path To Use

- Use [`steering-vectors-steering/`](steering-vectors-steering) if you want the smallest possible example. It runs a plain Hugging Face GPT-2 model, trains or loads one steering vector, and prints baseline vs steered output.
- Use [`easysteer-steering/`](easysteer-steering) if you want the more advanced setup. It uses EasySteer plus a custom `vllm` implementation, a CUDA GPU, and a precomputed `.gguf` steering vector.

## Repo Layout

```text
.
├── README.md
├── .python-version
├── pyproject.toml
├── steering-vectors-steering/
│   ├── steering-vectors-steering.py
│   └── steering_vector.pt
└── easysteer-steering/
    ├── README.md
    ├── pyproject.toml
    ├── easysteer-steering.py
    └── EasySteer/
```

Important top-level files and folders:

- [`pyproject.toml`](pyproject.toml): root metadata for the small `steering-vectors` demo. It targets Python 3.12 and lists `steering-vectors`, `transformers`, and `torch`.
- [`.python-version`](.python-version): root Python version pin, currently `3.12`.
- [`steering-vectors-steering/`](steering-vectors-steering): the lightweight demo and its cached steering vector.
- [`easysteer-steering/`](easysteer-steering): the EasySteer wrapper layer plus the upstream project checked out at [`EasySteer/`](easysteer-steering/EasySteer) as a submodule.

## Clone With Submodules

The EasySteer path depends on git submodules.

For a fresh clone, use:

```bash
git clone --recurse-submodules https://github.com/BurnyCoder/activation-vector-steering-llms-mechanistic-interpretability.git
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## `steering-vectors-steering`

This is the simplest example in the repo.

Files:

- [`steering-vectors-steering/steering-vectors-steering.py`](steering-vectors-steering/steering-vectors-steering.py): demo script
- [`steering-vectors-steering/steering_vector.pt`](steering-vectors-steering/steering_vector.pt): saved steering vector so the script can reuse it instead of retraining every run

### How It Works

The script:

1. loads `gpt2` with Hugging Face `transformers`
2. defines paired training examples with a positive answer and a negative answer
3. loads [`steering_vector.pt`](steering-vectors-steering/steering_vector.pt) if it already exists
4. otherwise trains a new steering vector with `train_steering_vector(...)` and saves it
5. generates one baseline completion and one steered completion for the same prompt
6. prints both so you can compare the effect of the vector

Because the script uses `Path("steering_vector.pt")`, run it from inside [`steering-vectors-steering/`](steering-vectors-steering) so the relative path resolves correctly.

### Environment For `steering-vectors-steering`

This path uses the root Python 3.12 toolchain, not the EasySteer environment.

You have two supported options:

#### Option A: reuse the shared parent environment at `../.venv`

In this checkout, the existing shared environment is `../.venv` relative to the repo root.

```bash
source ../.venv/bin/activate
uv pip install steering-vectors transformers torch
cd steering-vectors-steering
python steering-vectors-steering.py
```

#### Option B: create a new local environment in `./.venv`

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv sync
cd steering-vectors-steering
python steering-vectors-steering.py
```

If you rerun the script after [`steering_vector.pt`](steering-vectors-steering/steering_vector.pt) exists, it will reuse the saved vector instead of retraining it.

## `easysteer-steering`

This folder is a wrapper around an EasySteer submodule checkout.

Files:

- [`easysteer-steering/easysteer-steering.py`](easysteer-steering/easysteer-steering.py): wrapper demo script
- [`easysteer-steering/README.md`](easysteer-steering/README.md): the source of truth for running the wrapper
- [`easysteer-steering/pyproject.toml`](easysteer-steering/pyproject.toml): local `uv` project that points `vllm` at [`EasySteer/vllm-steer`](easysteer-steering/EasySteer/vllm-steer)
- [`easysteer-steering/EasySteer/`](easysteer-steering/EasySteer): EasySteer checked out as a submodule

### How The Wrapper Works

[`easysteer-steering.py`](easysteer-steering/easysteer-steering.py) is a convenience entrypoint so you can run the demo from [`easysteer-steering/`](easysteer-steering) instead of changing directories into `EasySteer/`.

It:

- resolves the checked-out [`EasySteer/`](easysteer-steering/EasySteer) directory relative to the wrapper
- prepends [`EasySteer/vllm-steer`](easysteer-steering/EasySteer/vllm-steer) to `sys.path`
- checks that at least one CUDA GPU is visible to PyTorch
- loads `Qwen/Qwen2.5-1.5B-Instruct` through `vllm` with steering enabled
- loads the bundled vector [`EasySteer/vectors/happy_diffmean.gguf`](easysteer-steering/EasySteer/vectors/happy_diffmean.gguf)
- runs one baseline generation with scale `0`
- runs a second generation with a "happy" steer at scale `5.0` across layers `10` through `25`
- prints both generations so you can compare the intervention

The wrapper defaults `CUDA_VISIBLE_DEVICES` to `0` when it is unset.

### Environment For `easysteer-steering`

Do not use the root Python 3.12 demo environment for this path.

This path uses the EasySteer-specific environment described in [`easysteer-steering/README.md`](easysteer-steering/README.md). In this checkout, that environment is the Python 3.10 virtual environment at [`easysteer-steering/EasySteer/.venv`](easysteer-steering/EasySteer/.venv).

Start with the wrapper README:

- [`easysteer-steering/README.md`](easysteer-steering/README.md)

The preferred run path once that environment is prepared is:

```bash
cd easysteer-steering
EasySteer/.venv/bin/python easysteer-steering.py
```

Useful runtime knobs supported by the wrapper:

- `CUDA_VISIBLE_DEVICES`: choose which GPU is visible; defaults to `0` if unset
- `EASYSTEER_GPU_MEMORY_UTILIZATION`: float in `(0, 1]` to override the automatic memory cap
- `EASYSTEER_MAX_MODEL_LEN`: `-1` for model default, or a positive integer

## Inside `EasySteer/`

[`easysteer-steering/EasySteer/`](easysteer-steering/EasySteer) is the heavy part of the repo. It is a checked-out upstream project referenced by git submodule, not just a few local files.

Major folders:

- [`easysteer-steering/EasySteer/easysteer/`](easysteer-steering/EasySteer/easysteer): main Python package for extraction, hidden-state capture, and steering utilities
- [`easysteer-steering/EasySteer/frontend/`](easysteer-steering/EasySteer/frontend): browser UI and backend helpers for interactive steering
- [`easysteer-steering/EasySteer/replications/`](easysteer-steering/EasySteer/replications): replication examples for papers and steering projects
- [`easysteer-steering/EasySteer/experiment/`](easysteer-steering/EasySteer/experiment): notebooks and artifacts for experiments, benchmarks, and evaluations
- [`easysteer-steering/EasySteer/vectors/`](easysteer-steering/EasySteer/vectors): precomputed steering vectors bundled with the project
- [`easysteer-steering/EasySteer/vllm-steer/`](easysteer-steering/EasySteer/vllm-steer): custom `vllm` source tree used by EasySteer
- [`easysteer-steering/EasySteer/hf-space/`](easysteer-steering/EasySteer/hf-space): Hugging Face Spaces app variant
- [`easysteer-steering/EasySteer/docker/`](easysteer-steering/EasySteer/docker): Docker build and test assets
- [`easysteer-steering/EasySteer/tests/`](easysteer-steering/EasySteer/tests): tests for demo entrypoints and frontend helpers
- [`easysteer-steering/EasySteer/figures/`](easysteer-steering/EasySteer/figures): images used by upstream docs and demos

### `easysteer/`

The core package is split into a few meaningful areas:

- [`easysteer-steering/EasySteer/easysteer/hidden_states/`](easysteer-steering/EasySteer/easysteer/hidden_states): wrappers for capturing hidden states from `vllm` during inference
- [`easysteer-steering/EasySteer/easysteer/steer/`](easysteer-steering/EasySteer/easysteer/steer): steering-vector extraction methods and shared interfaces
- [`easysteer-steering/EasySteer/easysteer/reft/`](easysteer-steering/EasySteer/easysteer/reft): ReFT-related demos and supporting code

Examples from the code:

- [`unified_interface.py`](easysteer-steering/EasySteer/easysteer/steer/unified_interface.py) exposes one extraction interface across methods such as DiffMean, PCA, LAT, and linear probes
- [`capture.py`](easysteer-steering/EasySteer/easysteer/hidden_states/capture.py) wraps `vllm` RPC calls to collect hidden states cleanly

### `frontend/`

The frontend is more than static HTML. It includes both a web UI and backend helper code.

- [`configs/`](easysteer-steering/EasySteer/frontend/configs): preset configs for chat, extraction, inference, multi-vector steering, and training
- [`core/`](easysteer-steering/EasySteer/frontend/core): GPU/resource helpers, prompt utilities, LLM management, and steering-request construction
- [`static/`](easysteer-steering/EasySteer/frontend/static): frontend assets such as CSS, JavaScript, and templates
- [`results/`](easysteer-steering/EasySteer/frontend/results): sample/demo outputs used by the UI
- [`app.py`](easysteer-steering/EasySteer/frontend/app.py) and related `*_api.py` files: API entrypoints for chat, inference, extraction, training, and SAE flows

### `replications/`

[`replications/`](easysteer-steering/EasySteer/replications) holds example reproductions and project-specific steering setups. Each subfolder usually documents one paper, benchmark, or steering case study.

Current subfolders include:

- [`bipo`](easysteer-steering/EasySteer/replications/bipo)
- [`cast`](easysteer-steering/EasySteer/replications/cast)
- [`controlingthinkingspeed`](easysteer-steering/EasySteer/replications/controlingthinkingspeed)
- [`creative_writing`](easysteer-steering/EasySteer/replications/creative_writing)
- [`fractreason`](easysteer-steering/EasySteer/replications/fractreason)
- [`improve_reasoning`](easysteer-steering/EasySteer/replications/improve_reasoning)
- [`lm_steer`](easysteer-steering/EasySteer/replications/lm_steer)
- [`loreft`](easysteer-steering/EasySteer/replications/loreft)
- [`refusal_direction`](easysteer-steering/EasySteer/replications/refusal_direction)
- [`sae_entities`](easysteer-steering/EasySteer/replications/sae_entities)
- [`sake`](easysteer-steering/EasySteer/replications/sake)
- [`seal`](easysteer-steering/EasySteer/replications/seal)
- [`sharp`](easysteer-steering/EasySteer/replications/sharp)
- [`steerable_chatbot`](easysteer-steering/EasySteer/replications/steerable_chatbot)

Many of these folders have their own `README.md` files.

### `experiment/`

[`experiment/`](easysteer-steering/EasySteer/experiment) contains notebooks and artifacts for internal evaluation work.

Current areas:

- [`efficiency/`](easysteer-steering/EasySteer/experiment/efficiency): throughput and batching comparisons
- [`hallucination/`](easysteer-steering/EasySteer/experiment/hallucination): data and notebooks around hallucination-style evaluation
- [`math/`](easysteer-steering/EasySteer/experiment/math): math steering experiments and saved vector artifacts

### `vectors/`

[`vectors/`](easysteer-steering/EasySteer/vectors) stores ready-to-use steering vectors shipped with the EasySteer checkout.

Examples currently present:

- [`happy_diffmean.gguf`](easysteer-steering/EasySteer/vectors/happy_diffmean.gguf)
- [`15973.pt`](easysteer-steering/EasySteer/vectors/15973.pt)
- [`2534.pt`](easysteer-steering/EasySteer/vectors/2534.pt)

### `vllm-steer/`

[`vllm-steer/`](easysteer-steering/EasySteer/vllm-steer) is the customized `vllm` source tree that makes the EasySteer path possible. It is large because it includes the engine code, docs, tests, benchmarks, Docker files, kernels, and examples expected from a full `vllm`-style project.

This is why the EasySteer side of the repo is much heavier than the minimal GPT-2 demo.

## Documentation Map

If you want more detail than this root README gives, use these docs next:

- [`easysteer-steering/README.md`](easysteer-steering/README.md): how to run the local EasySteer wrapper in this repo
- [`easysteer-steering/EasySteer/README.md`](easysteer-steering/EasySteer/README.md): upstream EasySteer project overview, installation, examples, and API usage
- [`easysteer-steering/EasySteer/frontend/README.md`](easysteer-steering/EasySteer/frontend/README.md): frontend/web UI overview
- [`easysteer-steering/EasySteer/replications/`](easysteer-steering/EasySteer/replications): individual replication READMEs for specific projects

## Quick Summary

- [`steering-vectors-steering/`](steering-vectors-steering) is the small CPU/GPU-friendly GPT-2 demo and should use either `../.venv` or a new local `./.venv` at the repo root.
- [`easysteer-steering/`](easysteer-steering) is the CUDA-heavy EasySteer path and should use the EasySteer-specific environment documented in [`easysteer-steering/README.md`](easysteer-steering/README.md).
- The root README explains how the workspace fits together. The subproject READMEs remain the detailed source of truth for their own setup and usage.
