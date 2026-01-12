# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

slime is an LLM post-training framework for RL scaling that connects Megatron (training) with SGLang (inference/rollout). It powers models like GLM-4.7, GLM-4.6, and GLM-4.5, and supports Qwen3, DeepSeek V3, and Llama 3 model families.

## Common Commands

### Installation
```bash
pip install -e .
```

### Run Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_chunked_gae.py

# Run tests by marker
pytest -m "unit"
pytest -m "integration"
```

### Code Style
```bash
# Install and run pre-commit hooks
apt install pre-commit -y
pre-commit install
pre-commit run --all-files --show-diff-on-failure --color=always
```

### Model Weight Conversion
```bash
# HuggingFace to Megatron format
PYTHONPATH=/path/to/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} --hf-checkpoint /path/to/hf --save /path/to/torch_dist

# Megatron to HuggingFace format
PYTHONPATH=/path/to/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    --input-dir /path/to/torch_dist/iter_xxx --output-dir /path/to/output --origin-hf-dir /path/to/hf
```

### Training
```bash
# Standard training (see scripts/ for model-specific examples)
bash scripts/run-glm4-9B.sh

# Async training
python train_async.py [args]

# Sync training
python train.py [args]
```

## Architecture

### Core Training Loop
The framework implements a "Data Sampling → Weight Update" loop:
1. **Rollout Phase**: SGLang generates responses for prompts
2. **Training Phase**: Megatron updates model weights
3. **Weight Sync**: Updated weights are transferred from Megatron to SGLang

### Key Modules

- **`slime/backends/megatron_utils/`**: Megatron training backend (actor, loss computation, checkpointing, weight conversion)
- **`slime/backends/fsdp_utils/`**: FSDP training backend alternative
- **`slime/backends/sglang_utils/`**: SGLang inference engine integration
- **`slime/rollout/`**: Data generation and reward computation
  - `sglang_rollout.py`: Main rollout generation logic with async generation
  - `rm_hub/`: Reward model implementations (deepscaler, f1, gpqa, math_utils)
  - `filter_hub/`: Dynamic sampling filters (e.g., check_reward_nonzero_std)
- **`slime/router/`**: Request routing between multiple SGLang servers
- **`slime/ray/`**: Ray-based distributed orchestration (placement groups, actor management)
- **`slime/utils/arguments.py`**: All slime-specific CLI arguments

### Training Backends
- `--train-backend megatron` (default): Uses Megatron-LM for training
- `--train-backend fsdp`: Uses PyTorch FSDP

### Colocated vs Disaggregated Mode
- **Disaggregated**: Training and inference on separate GPUs (default)
- **Colocated** (`--colocate`): Training and inference share the same GPUs with weight offloading

### Arguments Structure
1. **Megatron arguments**: Standard Megatron params (e.g., `--tensor-model-parallel-size`)
2. **SGLang arguments**: Prefixed with `--sglang-` (e.g., `--sglang-mem-fraction-static`)
3. **slime arguments**: See `slime/utils/arguments.py`

## Key Customization Points

- `--rollout-function-path`: Custom rollout generation function
- `--custom-generate-function-path`: Custom generation function for multi-turn/agent scenarios
- `--custom-rm-path`: Custom reward model function
- `--custom-loss-function-path`: Custom loss function
- `--dynamic-sampling-filter-path`: Custom filter for dynamic sampling

## Debugging

```bash
# Rollout only (no training)
--debug-rollout-only

# Training only (no SGLang)
--debug-train-only

# Save rollout data for debugging
--save-debug-rollout-data /path/to/data_{rollout_id}.pt

# Load saved rollout data
--load-debug-rollout-data /path/to/data_{rollout_id}.pt
```

## Model Configuration

Model-specific Megatron configs are in `scripts/models/`. Source the appropriate config before conversion or training:
```bash
source scripts/models/glm4-9B.sh
```
