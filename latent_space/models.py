import torch
import torch.nn as nn


class Encoder(nn.Module):
   

    def __init__(self, latent_dim: int = 16):
        """
        Initializes the Encoder model.

        Args:
            latent_dim: The dimensionality of the output latent space.
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Linear(64 * 7 * 7, latent_dim)

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
        features = self.conv(x)
        flattened = features.view(features.size(0), -1)
        latent_vector = self.fc(flattened)
        normalized_latent_vector = nn.functional.normalize(latent_vector, p=2, dim=1)
        return normalized_latent_vector


class InverseModel(nn.Module):
    
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
            nn.Linear(hidden_dim, self.num_actions),
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


