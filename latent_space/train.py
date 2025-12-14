"""
Main training script for the Encoder and Inverse Model.
(This file requires no changes).
"""

import torch
import torch.nn as nn
import torch.optim as optim

from data import create_dataloader
from models import Encoder, InverseModel

# -- Hyperparameters --
LATENT_DIM = 16
NUM_ACTIONS = 9
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 500
EXPERIENCE_PATH = "pacman_experience.pkl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    """Main training loop."""
    print(f"Using device: {DEVICE}")

    # 1. Initialize models
    encoder = Encoder(latent_dim=LATENT_DIM).to(DEVICE)
    inverse_model = InverseModel(latent_dim=LATENT_DIM, num_actions=NUM_ACTIONS).to(DEVICE)

    # 2. Create DataLoader
    dataloader = create_dataloader(experience_path=EXPERIENCE_PATH, batch_size=BATCH_SIZE)

    # 3. Define Loss and Optimizer
    params_to_optimize = list(encoder.parameters()) + list(inverse_model.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for i, (s_t, s_t1, a_t) in enumerate(dataloader):
            s_t, s_t1, a_t = s_t.to(DEVICE), s_t1.to(DEVICE), a_t.to(DEVICE)

            optimizer.zero_grad()

            z_t = encoder(s_t)
            z_t1 = encoder(s_t1)

            predicted_action_logits = inverse_model(z_t, z_t1)
            loss = criterion(predicted_action_logits, a_t)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"--- End of Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {avg_loss:.4f} ---")

    # 5. Save the trained models
    torch.save(encoder.state_dict(), "encoder_model_grayscale.pth")
    torch.save(inverse_model.state_dict(), "inverse_model_grayscale.pth")
    print("Training complete. Models saved.")


if __name__ == "__main__":
    train()

