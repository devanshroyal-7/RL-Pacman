# DQN Pacman Reinforcement Learning

A Deep Q-Network (DQN) implementation for training an AI agent to play Pacman using PyTorch and the Atari Learning Environment (ALE), with a latent space encoder for state representation learning.

## Overview

This project implements a DQN agent that learns to play Ms. Pacman through reinforcement learning. The agent uses experience replay, target networks, and epsilon-greedy exploration. Additionally, it includes a latent space encoder and inverse model that learn compressed representations of game states.

## Features

- Deep Q-Network (DQN) with convolutional neural networks
- Experience Replay for stable learning
- Target Network updates for improved stability
- Frame Stacking for temporal information
- Atari Preprocessing (frame warping, reward clipping, etc.)
- TensorBoard Logging for training visualization
- Episode Recording for gameplay analysis
- Checkpoint System for saving/loading trained models
- Latent Space Encoder and Inverse Model for state representation learning

## Requirements

- Python 3.8+
- PyTorch 2.4.1
- Gymnasium with Atari support
- OpenCV
- TensorBoard

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dqn-pacman.git
cd dqn-pacman
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
cd latent_space
pip install -r requirements.txt
```

## Usage

### Training DQN Agent

Start training the DQN agent:

```bash
cd pacman_agent
python train_pacman.py
```

The training script will:
- Create necessary directories (checkpoints/, runs/, episode_recordings/)
- Train for 2M frames with research-backed hyperparameters
- Save checkpoints every 10k frames
- Record episodes at milestones
- Log training metrics to TensorBoard

### Finding Best Checkpoint

Extract the best performing checkpoint:

```bash
python find_best_checkpoint.py
```

This creates `best_policy.pt` from the best checkpoint in the checkpoints/ directory.

### Evaluation

Evaluate a trained model:

```bash
python evaluate_agent.py --checkpoint checkpoints/dqn_pacman_research_XXXXX_avgXXX.pt --episodes 5
```

Options:
- `--checkpoint`: Path to checkpoint file
- `--episodes`: Number of episodes to evaluate (default: 5)
- `--no-render`: Disable rendering for faster evaluation

### Latent Space Training

1. Collect experience data using the trained policy:

```bash
cd latent_space
python collect_experience.py
```

This runs the trained policy for complete episodes and saves state transitions to `pacman_experience.pkl`.

2. Train the encoder and inverse model:

```bash
python train.py
```

This trains the encoder to compress 84x84 grayscale frames into 16-dimensional latent vectors and an inverse model to predict actions from state transitions.

### Running Policy with Latent Space

Run the trained policy with latent space loop detection:

```bash
python run_latent_policy.py --policy best_policy.pt --encoder latent_space/encoder_model_grayscale.pth --episodes 5
```

### Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir runs
```

## Architecture

### DQN Model
- Input: Stacked grayscale frames (84x84x4)
- Convolutional Layers: 3 conv layers with ReLU activation
- Fully Connected: 2 FC layers with 512 hidden units
- Output: Q-values for 9 possible actions (MsPacman action space)

### Environment Wrappers
- NoopReset: Random no-op actions on reset
- MaxAndSkip: Skip frames and take max over last 2 observations
- WarpFrame: Resize frames to 84x84 grayscale
- FrameStack: Stack 4 consecutive frames

### Latent Space Model
- Encoder: Convolutional network that compresses 84x84 grayscale images to 16-dimensional L2-normalized latent vectors
- Inverse Model: MLP that predicts actions from latent state transitions

## Hyperparameters

### DQN Training
- Buffer Size: 1,000,000
- Batch Size: 32
- Learning Rate: 2.5e-4 (RMSProp)
- Gamma: 0.99
- Epsilon: 1.0 → 0.01 (linear decay over 1M frames)
- Target Update: Every 1,000 steps
- Training Start: After 100,000 frames
- Max Frames: 2,000,000
- Frame Stack: 4 consecutive frames

### Latent Space Training
- Latent Dimension: 16
- Learning Rate: 1e-4 (Adam)
- Batch Size: 64
- Epochs: 500
- Number of Actions: 9

## Project Structure

```
├── pacman_agent/
│   ├── train_pacman.py          # Main DQN training script
│   ├── evaluate_agent.py        # Model evaluation script
│   ├── dqn_model.py            # DQN neural network architecture
│   ├── wrappers.py             # Environment preprocessing wrappers
│   └── replay_buffer.py        # Experience replay buffer
├── latent_space/
│   ├── collect_experience.py   # Collect experience using trained policy
│   ├── train.py                # Train encoder and inverse model
│   ├── models.py               # Encoder and inverse model architectures
│   ├── data.py                 # Dataset and DataLoader for experience data
│   └── requirements.txt        # Latent space dependencies
├── run_latent_policy.py        # Run policy with latent space loop detection
├── find_best_checkpoint.py     # Extract best checkpoint
├── requirements.txt            # Main project dependencies
├── checkpoints/                # Saved model checkpoints
├── runs/                       # TensorBoard logs
└── episode_recordings/         # Video recordings
```

## License

This project is open source and available under the MIT License.
