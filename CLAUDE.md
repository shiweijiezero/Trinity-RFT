# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Trinity-RFT is a framework for reinforcement fine-tuning (RFT) of large language models. The architecture is built around three decoupled components: **Explorer** (data generation), **Trainer** (model training), and **Buffer** (data storage), coordinated by a central **Synchronizer**.

## Development Setup

### Installation
```bash
# From source (recommended for development)
pip install -e ".[dev]"
pip install -e ".[flash_attn]"
# or: pip install flash-attn==2.8.1 --no-build-isolation
```

### Code Quality
```bash
# Run pre-commit checks before committing
pre-commit run --all-files

# Run tests
python -m pytest tests

# Run specific test
python -m pytest tests/buffer/file_test.py

# Run with coverage
python -m pytest tests --cov=trinity
```

### Running the Framework
```bash
# Start Ray cluster first
ray start --head

# Run training with a config file
trinity run --config examples/grpo_gsm8k/gsm8k.yaml

# Launch web UI for config management
trinity studio --port 8080

# Benchmark mode for quick experiments
python benchmark/bench.py gsm8k --model_path /path/to/model
```

## Architecture

### Core Components (Trinity)

**Explorer** (`trinity/explorer/`): Generates rollout experiences
- `explorer.py` - Main Explorer Ray actor coordinating exploration
- `workflow_runner.py` - Ray actor executing workflows with inference models
- `scheduler.py` - Manages workflow runner pools and task distribution
- Loads inference models via vLLM, runs workflows on tasks from buffer, writes experiences back to buffer

**Trainer** (`trinity/trainer/`): Trains models on collected experiences
- `trainer.py` - Main Trainer Ray actor
- `verl_trainer.py` - Integration with veRL training backend
- Samples experiences from buffer, performs training steps, manages checkpoints, synchronizes weights with explorer

**Buffer** (`trinity/buffer/`): Central data storage and retrieval
- Storage backends: FILE (local JSONL/Parquet), QUEUE (in-memory with replay), SQL (database)
- `reader/` and `writer/` - Storage-agnostic interfaces
- `operators/` - Data processing operators (filters, mappers)
- `pipelines/` - Experience and task data pipelines

**Synchronizer** (`trinity/manager/synchronizer.py`): Coordinates weight syncing
- Methods: NCCL (online synchronous), CHECKPOINT (offline), MEMORY (in-memory)
- Sync styles: FIXED (periodic), DYNAMIC_BY_TRAINER, DYNAMIC_BY_EXPLORER
- Tracks explorer/trainer running status and coordinates synchronization

### Key Abstractions

**Config System** (`trinity/common/config.py`):
- YAML-based hierarchical configuration loaded via OmegaConf
- Main sections: `mode`, `algorithm`, `model`, `buffer`, `explorer`, `trainer`, `synchronizer`, `monitor`
- Use `config.check_and_update()` to validate and auto-configure
- See `examples/` for reference configurations

**Workflows** (`trinity/common/workflows/`):
- Abstract execution pattern for generating experiences from tasks
- Key methods: `run()` for sync, `run_async()` for async execution
- Attributes: `is_async`, `can_reset`, `can_repeat`
- Built-in workflows: math, tool-calling, multi-turn, AgentScope integration

**Experience Data Structure** (`trinity/common/experience.py`):
- `Experience` - Single trajectory with tokens, logprobs, advantages, returns, metadata
- `Experiences` - Batch of experiences with tensor operations and padding
- `EID` - Unique identifier: `{batch_id}/{task_id}/{run_id}/{step_id}`

**Algorithms** (`trinity/algorithm/`):
- Supported: SFT, PPO, GRPO, DPO
- Algorithm-specific components: advantage_fn, policy_loss_fn, kl_fn, entropy_loss_fn, sample_strategy
- GRPO uses `repeat_times > 1` for multiple rollouts per task

### Directory Structure
```
trinity/
├── cli/              # Command-line interface (launcher.py main entry)
├── common/           # Shared abstractions (config, workflows, rewards, models)
├── explorer/         # Data generation component
├── trainer/          # Model training component
├── buffer/           # Data storage and pipelines
├── algorithm/        # RL algorithm implementations
├── manager/          # State management and synchronization
├── utils/            # Logging, monitoring, plugin system
└── plugins/          # Plugin system for extensibility
```

## Development Workflow

### Adding a New Workflow
1. Create class inheriting from `Workflow` in `trinity/common/workflows/`
2. Implement `run()` or `run_async()` method returning list of `Experience` objects
3. Set class attributes: `is_async`, `can_reset`, `can_repeat`
4. Register in `trinity/common/workflows/__init__.py`
5. Reference by module path in config YAML: `workflow: "trinity.common.workflows.your_workflow.YourWorkflow"`

### Adding a New Algorithm Component
1. Create implementation in appropriate `trinity/algorithm/` subdirectory
2. Use `@register_{component_type}()` decorator for auto-registration
3. Implement required interface (e.g., `AdvantageFunction`, `PolicyLossFunction`)
4. Reference by name in config YAML algorithm section

### Adding Buffer Operators
1. Create operator class in `trinity/buffer/operators/`
2. Inherit from `ExperienceOperator` or appropriate base
3. Implement `__call__()` method for data transformation
4. Add to experience pipeline in config under `data_processor.experience_pipeline`

### Writing Tests
- Tests located in `tests/` mirroring source structure
- Use `unittest.IsolatedAsyncioTestCase` for async tests
- Helper functions in `tests/tools.py`: `get_template_config()`, `get_unittest_dataset_config()`
- Run individual test files: `python -m pytest tests/path/to/test.py`

## Configuration Details

### Training Modes
- `explore` - Explorer only (data generation)
- `train` - Trainer only (offline training from existing data)
- `both` - Coordinated explorer + trainer with synchronization
- `bench` - Evaluation mode on eval tasksets
- `serve` - Explorer as API server

### Buffer Storage Configuration
```yaml
buffer:
  batch_size: 96  # Explorer reads this many tasks per batch
  train_batch_size: 48  # Trainer samples this many experiences
  explorer_input:
    taskset:
      name: "my_taskset"
      storage_type: "FILE"  # or QUEUE, SQL
      path: "/path/to/tasks.jsonl"
      total_epochs: 1  # or total_steps: N
  trainer_input:
    experience_buffer:
      name: "exp_buffer"
      storage_type: "QUEUE"  # Enables replay for off-policy
```

### Synchronization Configuration
```yaml
synchronizer:
  sync_method: "NCCL"  # or CHECKPOINT, MEMORY
  sync_style: "FIXED"  # or DYNAMIC_BY_TRAINER, DYNAMIC_BY_EXPLORER
  sync_interval: 2  # Sync every 2 trainer steps (FIXED) or iterations
```

### Multi-Stage Training
```yaml
# Root config references stages
stages:
  - stage1.yaml
  - stage2.yaml

# Each stage config has full training parameters
# StateManager automatically progresses through stages
```

## Important Implementation Notes

### Ray Actors and Async
- Explorer and Trainer are Ray remote actors (`@ray.remote`)
- Most internal methods use `async/await` for non-blocking I/O
- WorkflowRunner actors can be sync or async depending on workflow's `is_async` attribute
- Buffer writers typically async; readers can be sync or async

### Model Loading
- Models loaded via `trinity/common/models/model_wrapper.py`
- Inference uses vLLM for efficiency
- Training uses veRL (wraps PyTorch FSDP) or Megatron-LM
- Model paths can be HuggingFace identifiers or local paths

### Checkpoint Management
- Checkpoints saved to `checkpoint_root_dir/{project}/{name}/checkpoints/`
- StateManager tracks: `latest_iteration`, `latest_task_index`, `latest_exp_index`
- Enable recovery with `continue_from_checkpoint: true` in config
- Each stage in multi-stage training has separate checkpoints

### Plugin System
- Plugins loaded from `--plugin-dir` or `TRINITY_PLUGIN_DIR` env var
- Register custom components: workflows, reward functions, operators, monitors
- Use `@register_*()` decorators from `trinity/utils/registry.py`

### Monitoring and Logging
- Supports WandB and TensorBoard (configured in `monitor` section)
- Logs to console and file: `checkpoint_root_dir/{project}/{name}/logs/`
- Use `from trinity.utils.log import logger` for consistent logging

## Common Patterns

### Extending Reward Functions
```python
from trinity.common.rewards import RewardFn, register_reward_fn

@register_reward_fn()
class MyRewardFn(RewardFn):
    def compute_reward(self, experience, task):
        # Return float reward
        return reward_value
```

### Custom Data Processing
```python
from trinity.buffer.operators import ExperienceOperator

class MyFilter(ExperienceOperator):
    def __call__(self, experiences):
        # Filter and return list of experiences
        return [exp for exp in experiences if condition(exp)]
```

### Debugging Workflows
```bash
# Debug specific components without full training
trinity debug --module workflow
trinity debug --module inference_model
```

## Code Style

- Line length: 100 characters (Black formatter)
- Import sorting: isort with Black profile
- Type hints encouraged but not strictly enforced
- Docstrings: Google style (flake8-docstrings checks)
- Complexity limit: 15 (McCabe complexity)

## Key Dependencies

- **Ray**: Distributed computing framework
- **vLLM**: Fast LLM inference (0.9.1 to 0.10.2)
- **veRL**: Training backend (0.5.0)
- **transformers**: HuggingFace models
- **OmegaConf**: Configuration management
- **tensordict**: Experience data structures
- Optional: **Data-Juicer** (task pipelines), **AgentScope** (agentic workflows), **Megatron-Core** (large-scale training)

# 修改指南
- workflow存在：trinity/common/workflows/envs/R3L
- 配置存在：examples/R3L
- 4个环境：alfworld、dapo、scienceworld、webshop
- 5个算法：RAFT、GRPO、OPMD、OPMD_Reweight_Adv(只有配置和OPMD不同，workflow与OPMD相同)、OPMD_R3L
- trinity/common/workflows/__init__.py中注册workflow，需要和配置和workflow中的名称保持一致
