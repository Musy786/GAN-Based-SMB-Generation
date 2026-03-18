import os
import json
from train_gan import train_gan, generate_and_visualise

# Base data path
project_root = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(project_root, "data")
dataset_folders = ["ge", "pattern_count", "smb"]

# Training settings
z_dim = 100
epochs = 100

# Loop through each dataset and train separately
for folder in dataset_folders:
    print(f"\n----- Training on {folder} -----")
    data_path = os.path.join(base_path, folder)

    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")

    # Use correct legend for smb vs others
    if folder == "smb":
        legend_path = os.path.join(data_path, "legend.json")
    else:
        legend_path = os.path.join(base_path, "marioai_legend.json")

    if not os.path.isfile(legend_path):
        raise FileNotFoundError(f"Legend file not found: {legend_path}")

    # Load legend
    with open(legend_path, "r") as f:
        legend = json.load(f)

    # Index Mappings
    char_to_index = {char: idx for idx, char in enumerate(legend["tiles"].keys())}
    index_to_char = {v: k for k, v in char_to_index.items()}

    # Train and visualise
    G_model, index_map = train_gan(data_path=data_path, z_dim=z_dim, epochs=epochs, legend_path=legend_path)
    generate_and_visualise(G_model, z_dim=z_dim, tile_map=index_to_char)

