import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Needed for some setups


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Helper function to resolve paths relative to this script's location, ensuring it works regardless of the current working directory.
def resolve_project_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(PROJECT_ROOT, path_value)


# Dataset pre-processing
class SuperMarioDataset(Dataset):
    def __init__(self, level_dir, fixed_size=(16, 16), char_to_index=None):
        # Loads and preprocesses datasets
        self.samples = []
        self.fixed_size = fixed_size
        self.char_to_index = char_to_index
        self.load_levels(level_dir)
        self.vocab_size = len(self.char_to_index)

    def load_levels(self, level_dir):
        # Preprocessing the data
        # Going through a loop in the directory provided and then storing them in a sample
        for filename in os.listdir(level_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(level_dir, filename)
                with open(filepath, "r") as file:
                    all_lines = file.readlines()

                    level_lines = []
                    for line in all_lines:
                        stripped_line = line.strip()
                        line_chars = list(stripped_line)
                        level_lines.append(line_chars)

                    level_tensor = self.ascii_to_tensor(level_lines)

                    self.samples.append(level_tensor)

    def ascii_to_tensor(self, ascii_grid):
        # Converts ASCII level to tensor
        h, w = self.fixed_size
        tensor = torch.zeros((h, w), dtype=torch.long)

        for i in range(h):
            for j in range(w):
                if i < len(ascii_grid) and j < len(ascii_grid[0]):
                    char = ascii_grid[i][j]
                    if char in self.char_to_index:
                        tensor[i][j] = self.char_to_index[char]
                    else:
                        tensor[i][j] = 0  # Default value

        return tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# Generator
class Generator(nn.Module):
    def __init__(self, z_dim, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# Training Loop
def train_gan(data_path="data/smb", z_dim=100, epochs=100, batch_size=32, legend_path="data/legend.json"):
    data_path = resolve_project_path(data_path)
    legend_path = resolve_project_path(legend_path)

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")
    if not os.path.isfile(legend_path):
        raise FileNotFoundError(f"Legend file not found: {legend_path}")

    # Opens up legend - the key for the ASCII representation
    with open(legend_path, "r") as f:
        legend = json.load(f)

    # Create character to index mapping
    char_to_index = {}
    index = 0

    for char in legend["tiles"].keys():  # Go through each character in the legend
        char_to_index[char] = index  # Assign the current index to the character
        index += 1  # Increment the index for the next character

    # Reverse mapping from index to character for later use in visualisation
    index_to_char = {}
    for char, idx in char_to_index.items():
        index_to_char[idx] = char

    # To see if the GPU is available for faster performance
    # If so, CUDA will be used, if not CPU will be used
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load dataset, then use DataLoader to wrap the data samples into mini-batches for training loop
    dataset = SuperMarioDataset(data_path, fixed_size=(16, 16), char_to_index=char_to_index)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialising the generator and discriminator
    G = Generator(z_dim, out_channels=dataset.vocab_size).to(device)
    D = Discriminator(in_channels=dataset.vocab_size).to(device)

    # For the Binary Cross-Entropy loss function for both G and D
    loss_fn = nn.BCELoss()

    # The optimisers for both G and D
    opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # For logging purposes
    G_losses = []
    D_losses = []

    for epoch in range(1, epochs + 1):
        for real in loader:
            real = real.to(device)
            b_size = real.size(0)

            # One-hot encode the real tiles
            real_oh = F.one_hot(real, num_classes=dataset.vocab_size).permute(0, 3, 1, 2).float()

            # Generate fake data
            z = torch.randn(b_size, z_dim, 1, 1).to(device)
            fake = G(z)

            # Train D
            D_real = D(real_oh).view(-1)
            D_fake = D(fake.detach()).view(-1)
            loss_D = loss_fn(D_real, torch.ones_like(D_real)) + loss_fn(D_fake, torch.zeros_like(D_fake))
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # Train G
            D_fake = D(fake).view(-1)
            loss_G = loss_fn(D_fake, torch.ones_like(D_fake))
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            D_losses.append(loss_D.item())
            G_losses.append(loss_G.item())

        print(f"Epoch {epoch}/{epochs} | D Loss: {loss_D.item():.4f}, G Loss: {loss_G.item():.4f}")

    return G, index_to_char


# Visualisation
def generate_and_visualise(G, z_dim=100, tile_map=None):
    # Generates and displays a level
    # For portability, we import matplotlib here and raise an error if it's not available
    # This error happens sometimes due to the way some environments handle dependencies
    # So this way we can provide a clear message about how to resolve it
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for visualisation. Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    G.eval()

    device = next(G.parameters()).device
    z = torch.randn(1, z_dim, 1, 1).to(device)

    with torch.no_grad():
        output = G(z).squeeze(0).cpu().numpy()
        predicted = np.argmax(output, axis=0)

        if tile_map:
            print("\nASCII Representation:")
            for row in predicted:
                print("".join(tile_map.get(int(t), "?") for t in row))

        plt.imshow(predicted, cmap="tab20")
        plt.title("Generated Level (Discrete Tiles)")
        plt.colorbar()
        # Non-blocking show so the script can continue and exit without waiting for window close.
        plt.show(block=False)
        plt.pause(0.001)

# This allows the script to be run directly for training and visualisation, while also allowing the functions to be 
# imported and used in other contexts (like train_all.py).
def parse_args():
    parser = argparse.ArgumentParser(description="Train GAN for SMB level generation.")
    parser.add_argument("--data-path", default="data/smb", help="Path to dataset folder.")
    parser.add_argument("--legend-path", default="data/smb/legend.json", help="Path to legend JSON.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--z-dim", type=int, default=100, help="Latent vector size.")
    parser.add_argument("--no-vis", action="store_true", help="Skip matplotlib visualisation at the end.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    legend_path = resolve_project_path(args.legend_path)
    data_path = resolve_project_path(args.data_path)

    # Retrieve legend file
    with open(legend_path, "r") as f:
        legend = json.load(f)

    # Load in index mappings
    char_to_index = {char: idx for idx, char in enumerate(legend["tiles"].keys())}
    index_to_char = {v: k for k, v in char_to_index.items()}

    # Train the GAN
    G_model, index_map = train_gan(
        data_path=data_path,
        z_dim=args.z_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        legend_path=legend_path,
    )

    # Generate and visualise a level unless disabled for quick smoke tests
    if not args.no_vis:
        generate_and_visualise(G_model, z_dim=args.z_dim, tile_map=index_to_char)