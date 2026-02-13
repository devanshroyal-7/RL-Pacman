# Pacman Reinforcement Learning with Policy Oscillation elemination with RTHS

### Latent Space Model
- Encoder: Convolutional network that compresses 84x84 grayscale images to 16-dimensional L2-normalized latent vectors
- Inverse Model: MLP that predicts actions from latent state transitions

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
