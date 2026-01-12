# slime Architecture Guide

This document provides a comprehensive walkthrough of the slime codebase, covering all major components, their designs, and implementations.

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
   - [Entry Points](#entry-points)
   - [Ray Orchestration](#ray-orchestration)
   - [Training Backends](#training-backends)
   - [Rollout System](#rollout-system)
   - [Weight Synchronization](#weight-synchronization)
   - [Router Layer](#router-layer)
4. [Data Flow](#data-flow)
5. [Key Abstractions](#key-abstractions)

---

## Overview

slime is an LLM post-training framework designed for RL scaling. It connects:
- **Megatron-LM** for distributed training
- **SGLang** for fast inference/rollout generation

The framework supports multiple model architectures (GLM, Qwen3, DeepSeek V3, Llama 3) and training paradigms (PPO, GRPO, GSPO, REINFORCE++).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Ray Cluster                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────┐        ┌──────────────────────────────────────┐  │
│  │   RolloutManager     │◄──────►│           SlimeRouter                 │  │
│  │   (Orchestrator)     │        │   ┌─────────────────────────────┐    │  │
│  └──────────────────────┘        │   │      Middleware Stack       │    │  │
│           │                       │   │  (RadixTree, Caching, etc)  │    │  │
│           │                       │   └─────────────────────────────┘    │  │
│           ▼                       └──────────────────────────────────────┘  │
│  ┌──────────────────────┐                          │                        │
│  │   PlacementGroup     │                          ▼                        │
│  │   (GPU Allocation)   │        ┌──────────────────────────────────────┐  │
│  └──────────────────────┘        │         SGLang Engines (TP)          │  │
│           │                       │  ┌────────┐ ┌────────┐ ┌────────┐   │  │
│           ▼                       │  │Engine 1│ │Engine 2│ │Engine N│   │  │
│  ┌──────────────────────┐        │  └────────┘ └────────┘ └────────┘   │  │
│  │   RayTrainGroup      │        └──────────────────────────────────────┘  │
│  │  ┌────────────────┐  │                          ▲                        │
│  │  │ Actor Workers  │  │                          │ Weight Updates         │
│  │  │ (Megatron/FSDP)│──┼──────────────────────────┘                        │
│  │  └────────────────┘  │                                                   │
│  │  ┌────────────────┐  │                                                   │
│  │  │ Critic Workers │  │                                                   │
│  │  │  (Optional)    │  │                                                   │
│  │  └────────────────┘  │                                                   │
│  └──────────────────────┘                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### Entry Points

| File | Description |
|------|-------------|
| `train.py` | Synchronous training entry point |
| `train_async.py` | Asynchronous training entry point |

**Training Loop Flow:**

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Init Ray  │────►│  Create PG  │────►│ Init Actors │────►│  Main Loop  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                    ┌──────────────────────────────────────────────┘
                    ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                        Main Loop                              │
        │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────────┐  │
        │  │ Rollout │──►│  Train  │──►│  Sync   │──►│ Checkpoint  │  │
        │  │ (SGLang)│   │(Megatron)│   │ Weights │   │   (Save)    │  │
        │  └─────────┘   └─────────┘   └─────────┘   └─────────────┘  │
        │       ▲                                            │         │
        │       └────────────────────────────────────────────┘         │
        └─────────────────────────────────────────────────────────────┘
```

---

### Ray Orchestration

**Location:** `slime/ray/`

| Module | Purpose |
|--------|---------|
| `placement_group.py` | GPU resource allocation and scheduling |
| `actor_group.py` | Ray actor management for training workers |
| `rollout.py` | Rollout process coordination |
| `rollout_manager.py` | Manages SGLang engines lifecycle |

**PlacementGroup Strategy:**

```
┌─────────────────────────────────────────────────────────────────┐
│                     Disaggregated Mode                           │
├─────────────────────────────────────────────────────────────────┤
│  Node 0                    │  Node 1                            │
│  ┌───┬───┬───┬───┐        │  ┌───┬───┬───┬───┐                 │
│  │G0 │G1 │G2 │G3 │ Actor  │  │G0 │G1 │G2 │G3 │ Actor           │
│  ├───┼───┼───┼───┤        │  ├───┼───┼───┼───┤                 │
│  │G4 │G5 │G6 │G7 │ Rollout│  │G4 │G5 │G6 │G7 │ Rollout         │
│  └───┴───┴───┴───┘        │  └───┴───┴───┴───┘                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Colocated Mode                              │
├─────────────────────────────────────────────────────────────────┤
│  Node 0                    │  Node 1                            │
│  ┌───┬───┬───┬───┬───┬───┬───┬───┐                              │
│  │G0 │G1 │G2 │G3 │G4 │G5 │G6 │G7 │  Shared Actor+Rollout       │
│  └───┴───┴───┴───┴───┴───┴───┴───┘  (Time-multiplexed)          │
│                                                                   │
│  Phase 1: SGLang active (Model on GPU)                           │
│  Phase 2: Megatron active (SGLang offloaded to CPU)              │
└─────────────────────────────────────────────────────────────────┘
```

**RayTrainGroup Class:**

```python
class RayTrainGroup:
    """Manages a group of Ray actors for distributed training"""

    # Key methods:
    def __init__(self, args, num_nodes, num_gpus_per_node, pg, ...)
    def async_init(args, role, with_ref)     # Initialize workers
    def async_train(rollout_id, rollout_data_ref)  # Training step
    def update_weights()                      # Sync weights to SGLang
    def onload() / offload()                  # Memory management
    def save_model(rollout_id)                # Checkpointing
```

---

### Training Backends

**Location:** `slime/backends/`

```
slime/backends/
├── megatron_utils/          # Megatron-LM backend
│   ├── actor.py             # MegatronTrainRayActor
│   ├── loss.py              # Policy/Value/SFT loss functions
│   ├── model.py             # Model initialization, train step
│   ├── checkpoint.py        # Checkpoint save/load
│   ├── initialize.py        # Megatron initialization
│   └── update_weight/       # Weight sync to SGLang
│       ├── update_weight_from_distributed.py  # NCCL broadcast
│       └── update_weight_from_tensor.py       # IPC for colocated
│
├── fsdp_utils/              # PyTorch FSDP backend (alternative)
│
└── sglang_utils/            # SGLang inference engine
    ├── sglang_engine.py     # SGLang server wrapper
    └── arguments.py         # SGLang-specific arguments
```

#### Megatron Backend

**MegatronTrainRayActor** (`slime/backends/megatron_utils/actor.py`):

| Method | Description |
|--------|-------------|
| `init()` | Initialize model, optimizer, scheduler |
| `train(rollout_id, rollout_data_ref)` | Execute one training iteration |
| `train_actor()` | Actor model training logic |
| `train_critic()` | Critic model training logic |
| `update_weights()` | Broadcast weights to SGLang |
| `sleep() / wake_up()` | Offload/onload for colocated mode |

**Training Flow:**

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          train_actor() Flow                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐ │
│  │  Get Data   │──►│ Switch to   │──►│ Compute Ref │──►│  Switch to Old  │ │
│  │  Iterator   │   │ Ref Model   │   │  Log Probs  │   │   Actor Model   │ │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────────┘ │
│                                                                │            │
│        ┌───────────────────────────────────────────────────────┘            │
│        ▼                                                                    │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐   │
│  │ Compute Actor   │──►│    Compute      │──►│  Switch to Actor Model  │   │
│  │   Log Probs     │   │ Advantages/GAE  │   │    + Run Train Step     │   │
│  └─────────────────┘   └─────────────────┘   └─────────────────────────┘   │
│                                                                │            │
│        ┌───────────────────────────────────────────────────────┘            │
│        ▼                                                                    │
│  ┌─────────────────┐                                                        │
│  │  Backup Actor   │                                                        │
│  │    Weights      │                                                        │
│  └─────────────────┘                                                        │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

**Loss Functions** (`slime/backends/megatron_utils/loss.py`):

| Function | Use Case |
|----------|----------|
| `policy_loss_function()` | PPO/GRPO/GSPO policy gradient loss |
| `value_loss_function()` | Clipped value function loss |
| `sft_loss_function()` | Supervised fine-tuning NLL loss |
| `compute_advantages_and_returns()` | GAE/GRPO/REINFORCE++ advantage estimation |

**Advantage Estimators:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Advantage Estimation Options                      │
├───────────────────────┬─────────────────────────────────────────────┤
│  Estimator            │  Description                                 │
├───────────────────────┼─────────────────────────────────────────────┤
│  ppo                  │  GAE with value baseline                    │
│  grpo                 │  Group-relative advantages (no critic)      │
│  gspo                 │  Group sequence-level policy optimization   │
│  reinforce_plus_plus  │  REINFORCE with γ-discounted returns        │
│  reinforce_plus_plus_ │  REINFORCE with moving average baseline     │
│  baseline             │                                              │
│  on_policy_distillation│ Teacher-student log-prob difference        │
└───────────────────────┴─────────────────────────────────────────────┘
```

#### SGLang Backend

**SGLangEngine** (`slime/backends/sglang_utils/sglang_engine.py`):

```python
class SGLangEngine(RayActor):
    """Wrapper for SGLang inference server"""

    # Lifecycle
    def init(dist_init_addr, port, nccl_port, ...)
    def shutdown()

    # Weight Management
    def update_weights_from_tensor(...)      # Colocated mode
    def update_weights_from_distributed(...) # Disaggregated mode
    def init_weights_update_group(...)       # NCCL group setup

    # Memory Management
    def flush_cache()
    def release_memory_occupation()
    def resume_memory_occupation(tags)

    # Generation Control
    def pause_generation()
    def continue_generation()
```

---

### Rollout System

**Location:** `slime/rollout/`

```
slime/rollout/
├── sglang_rollout.py        # Main rollout generation logic
├── data_source.py           # Data source abstractions
├── base_types.py            # RolloutFnOutput types
├── rm_hub/                  # Reward model implementations
│   ├── math_utils.py        # Math problem verification
│   ├── deepscaler.py        # DeepScaler RM
│   ├── f1.py                # F1 score RM
│   ├── gpqa.py              # GPQA benchmark RM
│   └── ifbench.py           # IFBench RM
│
└── filter_hub/              # Dynamic sampling filters
    ├── base_types.py        # Filter interfaces
    └── dynamic_sampling_filters.py
```

**Rollout Generation Flow:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        generate_rollout_async()                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐                                                         │
│  │  DataSource     │ ◄─── RolloutDataSource / RolloutDataSourceWithBuffer   │
│  │  .get_samples() │                                                         │
│  └────────┬────────┘                                                         │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    GenerateState (Singleton)                         │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │    │
│  │  │  Semaphore  │  │ Tokenizer/  │  │     Sampling Params         │  │    │
│  │  │ (Concurrency)│  │  Processor  │  │  (temp, top_p, max_len)     │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              Async Generation Pipeline (per sample group)           │    │
│  │                                                                      │    │
│  │  ┌────────────────┐   ┌────────────────┐   ┌────────────────────┐  │    │
│  │  │  generate()    │──►│  async_rm()    │──►│  DynamicFilter     │  │    │
│  │  │ (HTTP→SGLang)  │   │ (Compute RM)   │   │  (Keep/Reject)     │  │    │
│  │  └────────────────┘   └────────────────┘   └────────────────────┘  │    │
│  │                                                                      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────┐                                                         │
│  │ RolloutFnOutput │ ◄─── samples + metrics                                 │
│  └─────────────────┘                                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Data Source Hierarchy:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DataSource (ABC)                             │
│  ├── get_samples(num_samples) → list[list[Sample]]                  │
│  ├── add_samples(samples)                                            │
│  ├── save(rollout_id)                                               │
│  └── load(rollout_id)                                               │
└─────────────────────────────────────────────────────────────────────┘
           │
           ├──────────────────────────────────────────┐
           ▼                                          ▼
┌─────────────────────────┐              ┌─────────────────────────────┐
│   RolloutDataSource     │              │ RolloutDataSourceWithBuffer │
│  (Read-only streaming)  │              │  (Supports partial rollout) │
│                         │              │  ┌──────────────────────┐   │
│  - Dataset iteration    │              │  │ Buffer for aborted   │   │
│  - Epoch management     │              │  │ samples (resume)     │   │
│  - Shuffle per epoch    │              │  └──────────────────────┘   │
└─────────────────────────┘              └─────────────────────────────┘
```

---

### Weight Synchronization

**Location:** `slime/backends/megatron_utils/update_weight/`

Two strategies for syncing Megatron weights to SGLang:

| Strategy | Class | Use Case |
|----------|-------|----------|
| Distributed | `UpdateWeightFromDistributed` | Disaggregated mode (separate GPUs) |
| Tensor (IPC) | `UpdateWeightFromTensor` | Colocated mode (shared GPUs) |

**Distributed Weight Update (NCCL):**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   UpdateWeightFromDistributed                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Megatron Workers (PP stages)              SGLang Engines                    │
│                                                                              │
│  ┌─────────────┐                          ┌─────────────────────────┐       │
│  │  PP Rank 0  │──────────────────────────►│    NCCL Group          │       │
│  │  (DP=TP=0)  │   slime-pp_0             │    "slime-pp_0"        │       │
│  └─────────────┘                          └─────────────────────────┘       │
│                                                       │                      │
│  ┌─────────────┐                                      ▼                      │
│  │  PP Rank 1  │──────────────────────────►┌─────────────────────────┐       │
│  │  (DP=TP=0)  │   slime-pp_1             │ Broadcast weights per PP│       │
│  └─────────────┘                          └─────────────────────────┘       │
│        ...                                                                   │
│                                                                              │
│  Flow: pause_gen → flush_cache → broadcast → continue_gen                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Colocated Weight Update (IPC):**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      UpdateWeightFromTensor                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Megatron Workers                          SGLang Engines (same GPUs)       │
│                                                                              │
│  ┌─────────────┐    CPU Serialize          ┌─────────────────────────┐      │
│  │   Worker    │──────────────────────────►│      Ray IPC            │      │
│  │  (GPU 0-N)  │    Gloo gather            │   (Zero-copy share)     │      │
│  └─────────────┘                           └───────────┬─────────────┘      │
│                                                        │                     │
│                                                        ▼                     │
│                                            ┌─────────────────────────┐      │
│                                            │    SGLang Engine        │      │
│                                            │ update_weights_from_    │      │
│                                            │ tensor(serialized_data) │      │
│                                            └─────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Router Layer

**Location:** `slime/router/`

```
slime/router/
├── router.py                # SlimeRouter (FastAPI-based)
└── middleware_hub/          # Middleware implementations
    └── radix_tree_middleware.py  # Prefix caching
```

**SlimeRouter Architecture:**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SlimeRouter                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Middleware Stack                              │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │    │
│  │  │ RadixTree       │  │  Custom         │  │  ...                 │  │    │
│  │  │ Middleware      │  │  Middleware     │  │                      │  │    │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Load Balancer                                  │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │  worker_request_counts: {url: active_count}                   │   │   │
│  │  │  dead_workers: set()  (quarantined)                           │   │   │
│  │  │  _use_url(): select min(active_count) from healthy workers    │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     SGLang Worker Pool                                  │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐               │ │
│  │  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  │ Worker N │               │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Background Tasks:                                                           │
│  - Health check loop (interval: rollout_health_check_interval)              │
│  - Dead worker quarantine (threshold: health_check_failure_threshold)       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/add_worker?url=...` | POST | Register new worker |
| `/list_workers` | GET | List all registered workers |
| `/retrieve_from_text` | POST | RadixTree prefix lookup |
| `/{path}` | * | Proxy to SGLang workers |

---

## Data Flow

**Complete Training Iteration:**

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           Single Training Iteration                             │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. ROLLOUT PHASE                                                               │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  DataSource ─────► generate_rollout() ─────► SGLang Router ─────► Workers │  │
│  │                           │                                                │  │
│  │                           ▼                                                │  │
│  │                    ┌──────────────┐                                        │  │
│  │                    │   Sample     │  prompt + response + logprobs          │  │
│  │                    │   Groups     │  + reward + metadata                   │  │
│  │                    └──────────────┘                                        │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│  2. DATA PROCESSING                                                             │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  process_rollout_data() ─────► RolloutBatch (sharded by DP rank)         │  │
│  │  - tokens, loss_masks, rewards, rollout_log_probs                        │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│  3. TRAINING PHASE                                                              │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  MegatronTrainRayActor.train()                                            │  │
│  │  ┌─────────────────────────────────────────────────────────────────────┐ │  │
│  │  │  switch(ref) → compute_log_prob → switch(old_actor) → compute_log   │ │  │
│  │  │  → compute_advantages_and_returns → switch(actor) → train_step      │ │  │
│  │  └─────────────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│  4. WEIGHT SYNC                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  update_weights() ─────► pause ─────► broadcast ─────► continue          │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                      │                                          │
│                                      ▼                                          │
│  5. CHECKPOINT (periodic)                                                       │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  save_model() ─────► Megatron checkpoint + optional HF export            │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Abstractions

### Sample Type

```python
@dataclass
class Sample:
    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"

    # Core fields
    prompt: str | list[dict]           # Input prompt
    response: str = ""                 # Generated response
    tokens: list[int] = None           # Token IDs
    response_length: int = 0           # Response token count

    # Training data
    reward: float | dict = None        # Computed reward
    rollout_log_probs: list[float]     # Log probs during generation
    loss_mask: list[int] = None        # Mask for loss computation

    # Metadata
    index: int = -1                    # Global sample index
    group_index: int = -1              # Group index for GRPO
    label: Any = None                  # Ground truth for RM
    metadata: dict = None              # Custom metadata
    status: Status = Status.PENDING
```

### RolloutBatch Type

```python
RolloutBatch = TypedDict('RolloutBatch', {
    'tokens': list[torch.Tensor],
    'loss_masks': list[torch.Tensor],
    'rewards': list[float],
    'total_lengths': list[int],
    'response_lengths': list[int],
    'rollout_log_probs': list[torch.Tensor],  # Optional
    'ref_log_probs': list[torch.Tensor],      # Optional
    'advantages': list[torch.Tensor],         # Computed
    'returns': list[torch.Tensor],            # Computed
})
```

### Argument Categories

| Category | Prefix | Location |
|----------|--------|----------|
| Cluster | `--actor-*`, `--critic-*`, `--rollout-*` | `slime/utils/arguments.py` |
| Training | `--train-*`, `--qkv-format` | `slime/utils/arguments.py` |
| SGLang | `--sglang-*` | `slime/backends/sglang_utils/arguments.py` |
| Megatron | Standard Megatron args | Megatron-LM |
| Rollout | `--rollout-*` | `slime/utils/arguments.py` |
| RL | `--advantage-estimator`, `--eps-clip`, etc. | `slime/utils/arguments.py` |

---

## Extension Points

| Extension | Argument | Signature |
|-----------|----------|-----------|
| Custom Rollout | `--rollout-function-path` | `fn(args, rollout_id, data_source) → RolloutFnOutput` |
| Custom Generate | `--custom-generate-function-path` | `async fn(args, sample, params) → Sample` |
| Custom RM | `--custom-rm-path` | `async fn(args, sample) → reward` |
| Custom Loss | `--custom-loss-function-path` | `fn(args, batch, logits, reducer) → (loss, metrics)` |
| Custom TIS | `--custom-tis-function-path` | `fn(args, pg_loss, ...) → (loss, masks, metrics)` |
| Dynamic Filter | `--dynamic-sampling-filter-path` | `fn(args, group) → DynamicFilterOutput` |
| Buffer Filter | `--buffer-filter-path` | `fn(args, rollout_id, buffer, n) → samples` |

---

## Performance Considerations

1. **Memory Management**
   - Colocated mode uses `torch_memory_saver` for offloading
   - `--offload-train` and `--offload-rollout` control memory phases
   - `TensorBackuper` manages CPU copies for model switching

2. **Parallelism Configuration**
   - TP (Tensor Parallel): Within SGLang engine
   - PP (Pipeline Parallel): Across Megatron stages
   - DP (Data Parallel): Across training workers
   - CP (Context Parallel): For long sequences

3. **Weight Update Optimization**
   - Bucketed updates reduce NCCL calls
   - Expert parameters use separate EP groups
   - Pause/continue generation minimizes stale KV cache
