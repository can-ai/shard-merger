import json
import os
from safetensors.torch import load_file, save_file
import torch
import gc

# Paths
model_dir = os.getcwd()  # Use the current working directory
index_file = "diffusion_pytorch_model.safetensors.index.json"
output_file = "../Wan2.1-VACE-14B.safetensors"

# Verify index file exists
index_path = os.path.join(model_dir, index_file)
if not os.path.exists(index_path):
    raise FileNotFoundError(f"Index file not found: {index_path}")

# Load the index file
with open(index_path, "r") as f:
    index = json.load(f)

# Initialize an empty state dictionary for merged weights
merged_state_dict = {}

# Get unique shard files from the weight map
weight_map = index["weight_map"]
shard_files = sorted(set(weight_map.values()))  # Sort for consistency

# Load each shard and merge into the state dictionary
for shard_file in shard_files:
    print(f"Loading shard: {shard_file}")
    shard_path = os.path.join(model_dir, shard_file)
    if not os.path.exists(shard_path):
        raise FileNotFoundError(f"Shard file not found: {shard_path}")
    shard_state_dict = load_file(shard_path)
    merged_state_dict.update(shard_state_dict)
    # Free memory after loading each shard
    del shard_state_dict
    torch.cuda.empty_cache()  # Clear GPU memory if used
    gc.collect()  # Force garbage collection

# Save the merged model
print(f"Saving merged model to: {output_file}")
save_file(merged_state_dict, output_file)
print("Merge completed successfully!")
