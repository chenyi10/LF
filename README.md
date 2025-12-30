# Leader-Follower Multi-Hypothesis Navigation System

A complete reinforcement learning system for Leader-Follower cluster navigation with multi-hypothesis guidance, multi-head attention, and robust obstacle avoidance.

## Features

- **Multi-Hypothesis Guidance**: Leader generates K guidance directions with confidence weights
- **Multi-Head Attention (MHA)**: Leader uses MHA to process multi-view observations
- **Real Obstacle Avoidance**: Ray-based repulsion potential field + velocity projection
- **Formation Control**: PD controller for maintaining formation
- **Staged Training**: Stage 1 (Leader) → Stage 2 (Follower)
- **Gymnasium + Stable-Baselines3**: Industry-standard RL framework

## System Architecture

### Environment
- **2D continuous navigation** with static obstacles (corridors, corners, forks)
- **1 Leader + N-1 Followers** (default: 5 agents total)
- **Goal**: Navigate from start to goal without collisions

### Control Structure
For each agent i, the final action is:
```
a_i = a_i^RL + a_i^avoid + a_i^form
```
where:
- `a_i^RL`: Action from RL policy (learned)
- `a_i^avoid`: Obstacle avoidance (ray-based repulsion + velocity projection)
- `a_i^form`: Formation control (PD controller)

### Leader Network
- **Input**: Multi-view rays, goal vector, group summary
- **Architecture**: Multi-head attention transformer
- **Output**: K guidance directions {g₁, ..., gₖ} with confidence weights {w₁, ..., wₖ}

### Follower Network
- **Input**: Local observations, guidance vectors and weights
- **Architecture**: MLP with layer normalization
- **Output**: RL action a_i^RL

## Installation

```bash
# Clone repository
git clone https://github.com/chenyi10/LF.git
cd LF

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train Leader (Stage 1)
```bash
python train_leader.py --config config/default.yaml --output-dir models/leader --total-timesteps 500000
```

### 2. Train Follower (Stage 2)
```bash
python train_follower.py --config config/default.yaml --output-dir models/follower --leader-model models/leader/final_model --total-timesteps 500000
```

### 3. Evaluate
```bash
python eval.py --config config/default.yaml --follower-model models/follower/final_model --n-episodes 50 --output-dir results
```

### 4. Compare Configurations
```bash
python eval.py --config config/default.yaml --compare --output-dir results
```

## Project Structure

```
lf_mha_multi/
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml          # Configuration file
├── lf_mha_multi/
│   ├── envs/
│   │   ├── nav_env.py        # Main navigation environment
│   │   ├── obstacles.py      # Obstacle definitions and collision detection
│   │   ├── sensors.py        # Ray-based sensors
│   │   └── reward.py         # Reward calculation
│   ├── controllers/
│   │   ├── obstacle_avoid.py # Obstacle avoidance controller
│   │   └── formation_pd.py   # Formation PD controller
│   ├── models/
│   │   ├── leader_mha.py     # Leader policy with MHA
│   │   └── follower_policy.py # Follower policy
│   └── utils/
│       ├── config.py         # Configuration utilities
│       ├── seeding.py        # Random seed management
│       └── render.py         # Visualization utilities
├── train_leader.py           # Leader training script
├── train_follower.py         # Follower training script
└── eval.py                   # Evaluation script
```

## Configuration

Key parameters in `config/default.yaml`:

### Obstacle Avoidance
- `d_safe`: Safety distance threshold (default: 1.0)
- `k_avoid`: Repulsion strength (default: 0.15)
- `a_avoid_max`: Maximum avoidance action (default: 0.5)
- `d_proj`: Velocity projection threshold (default: 0.8)

### Formation Control
- `k_p`: Position gain (default: 0.5)
- `k_d`: Velocity damping (default: 0.3)
- `a_form_max`: Maximum formation action (default: 0.4)

### Leader Network
- `n_hypotheses`: Number of guidance directions K (default: 3)
- `use_mha`: Enable multi-head attention (default: true)
- `n_heads`: Number of attention heads (default: 4)
- `d_model`: Model dimension (default: 128)

## Obstacle Avoidance Implementation

### 1. Ray-Based Repulsion
For each ray with distance d < d_safe:
```python
magnitude = k_avoid * (1/d - 1/d_safe) / d²
direction = -ray_direction
contribution = magnitude * direction
```

### 2. Velocity Projection
Prevents agents from penetrating obstacles:
```python
if d < d_proj and v·n < 0:
    v ← v - (v·n) * n
```

## Training Stages

### Stage 1: Leader Training
- **Frozen**: Follower policy (uses fixed guidance following)
- **Learning**: Leader generates guidance {gₖ, wₖ}
- **Objective**: Maximize group progress to goal

### Stage 2: Follower Training
- **Frozen**: Leader policy (guidance fixed)
- **Learning**: Followers learn actions a_i^RL
- **Objective**: Follow guidance + avoid obstacles + maintain formation

## Evaluation Metrics

- **Success Rate**: Percentage of episodes reaching goal
- **Collision Rate**: Percentage of episodes with collisions
- **Average Steps**: Mean episode length
- **Guidance Diversity**: Average pairwise angle between guidance vectors

## Comparison Experiments

The system supports comparing:
- **K=1 vs K=3**: Single vs multi-hypothesis guidance
- **MHA vs MLP**: Attention-based vs feedforward leader

## Citation

If you use this code, please cite:

```bibtex
@software{lf_mha_multi2025,
  title={Leader-Follower Multi-Hypothesis Navigation System},
  author={chenyi10},
  year={2025},
  url={https://github.com/chenyi10/LF}
}
```

## License

MIT License

## Acknowledgments

- Built with [Gymnasium](https://gymnasium.farama.org/)
- RL training with [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- Inspired by multi-agent coordination and attention mechanisms research