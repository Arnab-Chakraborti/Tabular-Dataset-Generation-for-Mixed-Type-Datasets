import os
import yaml
import itertools
import copy

# 1. Define your static base configuration
base_config = {
    "vae_training": {
        "VAE epochs": 1000,
        "batch_size": 256,
        "latent_dim": 64,  
        "MMD Weight": 0.001,
        "dropout": 0.0,
        "lr": 0.0007
    },
    "architecture": {
        "Encoder_dims": [512, 256, 128],
        "Decoder_dims": [128, 256, 512],
        "hidden_dims": [512,256],
        "diffusion_hidden_dims": [512,256],
        "time_embedding_dim": 128,
        "activation": "SiLU",
        "use_layernorm": True
        
    },
    "diffusion": {
        "prediction_type": "noise",
        "num_diffusion_steps": 10,
        "diffusion_lr": 0.0007,
        "diffusion_epochs": 1000,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "schedule": "linear"
    }
}

# Param Grid
param_grid = {
    "latent_dim": [32, 64, 128, 192, 256],
    "MMD Weight": [20, 40, 60, 80, 100, 120, 140, 160],
    "num_diffusion_steps": [50, 100, 200, 300, 400, 500]
    
}

os.makedirs("configs", exist_ok=True)
# Generate all combinations for overnight run
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Write to YAML files
for i, combo in enumerate(combinations, start=1):
    current_config = copy.deepcopy(base_config)
    
    current_config["vae_training"]["latent_dim"] = combo["latent_dim"]
    current_config["vae_training"]["MMD Weight"] = combo["MMD Weight"]
    current_config["diffusion"]["num_diffusion_steps"] = combo["num_diffusion_steps"]
    
    # Save to file
    filename = f"configs/exp_{i}.yaml"
    with open(filename, 'w') as outfile:
        yaml.dump(current_config, outfile, default_flow_style=False, sort_keys=False)
        
    print(f"Generated {filename} -> Latent: {combo['latent_dim']} | MMD: {combo['MMD Weight']}")

print(f"\nSuccessfully generated {len(combinations)} YAML configurations.")
