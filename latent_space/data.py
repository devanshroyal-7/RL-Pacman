"""
This file contains the PyTorch Dataset and DataLoader for loading pre-processed
84x84x1 Pacman experience data.
"""

import pickle
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
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
        with open(experience_path, "rb") as f:
            self.experience = pickle.load(f)

        print(f"Loaded {len(self.experience)} samples.")

        # --- MODIFIED TRANSFORM FOR 1-CHANNEL DATA ---
        # The images are already 84x84 grayscale. We just need to convert to
        # a tensor, add a channel dimension, and normalize.
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Converts numpy (H, W) to (1, H, W) tensor and scales to [0,1]
                transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize for single channel
            ]
        )

    def __len__(self) -> int:
        return len(self.experience)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def create_dataloader(
    experience_path: str, batch_size: int, shuffle: bool = True, num_workers: int = 4
) -> DataLoader:
    """
    Creates a DataLoader for the Pacman experience dataset.
    """
    dataset = PacmanExperienceDataset(experience_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


