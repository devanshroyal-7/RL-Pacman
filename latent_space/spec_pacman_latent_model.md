# Goal: Create and Train a Latent Space Model for Pacman (Grayscale)

The goal is to implement and train a world model that learns a latent representation of Pacman game states, which are single-channel (grayscale) 84x84 images. The model consists of an Encoder and an Inverse Model. The Encoder, based on a modified ResNet-18, compresses states `s_t` and `s_t+1` into latent vectors `z_t` and `z_t+1`. The Inverse Model (an MLP) then predicts the action `a_t` taken to transition from `z_t` to `z_t+1`.

## Plan

1.  **`models.py`**: Define the `Encoder` and `InverseModel`. The `Encoder` will use a ResNet-18 architecture adapted to accept a 1-channel input instead of the standard 3.

2.  **`data.py`**: Create a PyTorch `Dataset` and `DataLoader` to handle batches of pre-processed `(s_t, s_t+1, a_t)` tuples. The transformation will be simpler as the data is already in the correct grayscale format.

3.  **`train.py`**: The main training script. This file requires minimal changes.

4.  **`collect_experience.py`**: Update the data collection script to use Gymnasium wrappers to automatically convert observations to grayscale and resize them to 84x84 before saving.

---

## File: `models.py`

```

"""

This file defines the neural network architectures for the Encoder and the Inverse Model,

adapted for single-channel (grayscale) input images.

"""

import torch

import torch.nn as nn

import torchvision.models as models

class Encoder(nn.Module):

    """

    Encoder network to compress an 84x84 grayscale image into a 16-dimensional latent vector.

    The architecture uses a pretrained ResNet-18, but its first convolutional layer is

    modified to accept 1 input channel instead of 3.

    """

    def __init__(self, latent_dim: int = 16):

        """

        Initializes the Encoder model.

        Args:

            latent_dim: The dimensionality of the output latent space.

        """

        super(Encoder, self).__init__()

        # Load a pretrained ResNet-18 model

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # --- MODIFICATION FOR 1-CHANNEL INPUT ---

        # Get the weights of the original first convolutional layer

        original_conv1_weights = self.resnet.conv1.weight.data

        # Create a new convolutional layer that accepts 1 channel

        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Adapt the pretrained weights from 3 channels to 1.

        # We do this by averaging the weights across the RGB channels.

        self.resnet.conv1.weight.data = original_conv1_weights.mean(dim=1, keepdim=True)

        # --- END MODIFICATION ---

        # The input to the original ResNet FC layer

        num_ftrs = self.resnet.fc.in_features

        # Replace the final fully connected layer with a new one for our latent space

        self.resnet.fc = nn.Linear(num_ftrs, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """

        Performs a forward pass through the encoder.

        Args:

            x: A batch of input images with shape (batch_size, 1, 84, 84).

               Images should be normalized.

        Returns:

            A batch of latent vectors with shape (batch_size, latent_dim),

            L2-normalized.

        """

        # Pass input through the modified ResNet backbone

        latent_vector = self.resnet(x)

        # Apply L2 normalization to constrain the embeddings to a unit hypersphere

        normalized_latent_vector = nn.functional.normalize(latent_vector, p=2, dim=1)

        return normalized_latent_vector

class InverseModel(nn.Module):

    """

    Inverse Model to predict the action taken between two latent states.

    (This class does not need changes as it operates on the latent vectors).

    """

    def __init__(self, latent_dim: int = 16, num_actions: int = 9, hidden_dim: int = 32):

        """

        Initializes the Inverse Model.

        Args:

            latent_dim: The dimensionality of the latent space.

            num_actions: The number of possible actions in the environment.

            hidden_dim: The number of neurons in the hidden layer.

        """

        super(InverseModel, self).__init__()

        self.input_dim = latent_dim * 2

        self.num_actions = num_actions

        self.network = nn.Sequential(

            nn.Linear(self.input_dim, hidden_dim),

            nn.LayerNorm(hidden_dim),

            nn.ReLU(),

            nn.Linear(hidden_dim, self.num_actions)

        )

    def forward(self, z_t: torch.Tensor, z_t1: torch.Tensor) -> torch.Tensor:

        """

        Performs a forward pass to predict action logits.

        Args:

            z_t: Latent vector for state t. Shape: (batch_size, latent_dim)

            z_t1: Latent vector for state t+1. Shape: (batch_size, latent_dim)

        Returns:

            Action logits. Shape: (batch_size, num_actions)

        """

        combined_z = torch.cat((z_t, z_t1), dim=1)

        action_logits = self.network(combined_z)

        return action_logits

```

---

## File: `data.py`

```

"""

This file contains the PyTorch Dataset and DataLoader for loading pre-processed

84x84x1 Pacman experience data.

"""

import torch

import pickle

import numpy as np

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

class PacmanExperienceDataset(Dataset):

    """

    A PyTorch Dataset for grayscale (s_t, s_t+1, a_t) tuples from Pacman.

    """

    def __init__(self, experience_path: str):

        """

        Args:

            experience_path: Path to the pickled file containing the list of

                             (state, next_state, action) tuples. States are

                             expected to be pre-processed (84, 84) numpy arrays.

        """

        print(f"Loading experience from {experience_path}...")

        with open(experience_path, 'rb') as f:

            self.experience = pickle.load(f)

        print(f"Loaded {len(self.experience)} samples.")

        # --- MODIFIED TRANSFORM FOR 1-CHANNEL DATA ---

        # The images are already 84x84 grayscale. We just need to convert to

        # a tensor, add a channel dimension, and normalize.

        self.transform = transforms.Compose([

            transforms.ToTensor(), # Converts numpy (H, W) to (1, H, W) tensor and scales to[10]

            transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize for single channel

        ])

    def __len__(self) -> int:

        return len(self.experience)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """

        Retrieves a sample from the dataset.

        """

        state_t, state_t1, action_t = self.experience[idx]

        # Apply transformations

        state_t = self.transform(state_t)

        state_t1 = self.transform(state_t1)

        # Convert action to a tensor

        action_t = torch.tensor(action_t, dtype=torch.long)

        return state_t, state_t1, action_t

def create_dataloader(experience_path: str, batch_size: int, shuffle: bool = True, num_workers: int = 4) -> DataLoader:

    """

    Creates a DataLoader for the Pacman experience dataset.

    """

    dataset = PacmanExperienceDataset(experience_path)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader

```

---

## File: `train.py`

```

"""

Main training script for the Encoder and Inverse Model.

(This file requires no changes).

"""

import torch

import torch.nn as nn

import torch.optim as optim

from models import Encoder, InverseModel

from data import create_dataloader

# -- Hyperparameters --

LATENT_DIM = 16

NUM_ACTIONS = 9

LEARNING_RATE = 1e-4

BATCH_SIZE = 64

NUM_EPOCHS = 50

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

```

---

## File: `collect_experience.py`

```

"""

Script to collect pre-processed 84x84 grayscale experience data from MsPacman.

"""

import gymnasium as gym

from gymnasium.wrappers import GrayScaleObservation, ResizeObservation

import pickle

from tqdm import tqdm

# -- Configuration --

ENV_NAME = "ALE/MsPacman-v5"

NUM_SAMPLES = 20000

OUTPUT_PATH = "pacman_experience.pkl"

IMAGE_SIZE = 84

def collect_random_experience():

    """

    Runs a random agent in the environment to collect state transitions.

    Uses Gymnasium wrappers to automatically pre-process observations.

    """

    print(f"Initializing environment: {ENV_NAME}")

    # Make the environment and apply wrappers

    env = gym.make(ENV_NAME, render_mode="rgb_array")

    # 1. Resize observations to 84x84

    env = ResizeObservation(env, shape=IMAGE_SIZE)

    # 2. Convert observations to grayscale (resulting shape is (84, 84))

    env = GrayScaleObservation(env, keep_dim=False) # keep_dim=False gives (H, W)

    experience_buffer = []

    print(f"Collecting {NUM_SAMPLES} pre-processed samples...")

    state, info = env.reset() # State is now a numpy array (84, 84)

    for _ in tqdm(range(NUM_SAMPLES)):

        action = env.action_space.sample()

        next_state, reward, terminated, truncated, info = env.step(action)

        experience_buffer.append((state.copy(), next_state.copy(), action))

        if terminated or truncated:

            state, info = env.reset()

        else:

            state = next_state

    env.close()

    print(f"\nSaving {len(experience_buffer)} samples to {OUTPUT_PATH}...")

    with open(OUTPUT_PATH, 'wb') as f:

        pickle.dump(experience_buffer, f)

    print("Experience collection complete.")

if __name__ == "__main__":

    collect_random_experience()

```

```

