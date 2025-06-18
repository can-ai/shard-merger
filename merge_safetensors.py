import json
import os
from safetensors.torch import load_file, save_file

# Paths
index_file = "diffusion_pytorch_model.safetensors.index.json"
output_file = "../Wan2.1-VACE-14B.safetensors"
model_dir = "app/ComfyUI/models/diffusion_models/wan_14b"  # Directory containing the shard files and index file

# Load the index file
with open(os.path.join(model_dir, index_file), "r") as f:
    index = json.load(f)

# Initialize an empty state dictionary for merged weights
merged_state_dict = {}

# Get unique shard files from the weight map
weight_map = index["weight_map"]
shard_files = set(weight_map.values())

# Load each shard and merge into the state dictionary
for shard_file in shard_files:
    print(f"Loading shard: {shard_file}")
    shard_path = os.path.join(model_dir, shard_file)
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Shard file {shard_path} not found!")
    shard_state_dict = load_file(shard_path)
    merged_state_dict.update(shard_state_dict)

# Save the merged model
print(f"Saving merged model to: {output_file}")
save_file(merged_state_dict, output_file)
print("Merge completed successfully!")
